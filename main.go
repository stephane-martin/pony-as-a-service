package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/gliderlabs/ssh"
	"github.com/joho/godotenv"
	"github.com/muesli/reflow/wordwrap"
	"github.com/sashabaranov/go-openai"
	"github.com/urfave/cli"
	"golang.org/x/term"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

func main() {
	_ = godotenv.Load()

	app := cli.NewApp()
	app.Name = "pony-as-a-service"
	app.Usage = "deliver ponies as a service"
	app.Action = serve
	app.Flags = []cli.Flag{
		cli.StringFlag{
			Name:   "ssh-addr",
			Usage:  "SSH listen address",
			Value:  "127.0.0.1:2222",
			EnvVar: "PONY_SSH_ADDR",
		},
		cli.StringFlag{
			Name:   "hostkey",
			Usage:  "SSH hostkey's file path",
			Value:  "ponyhost",
			EnvVar: "PONY_HOSTKEY",
		},
		cli.StringFlag{
			Name:     "openai-api-key",
			Usage:    "OpenAI API key",
			EnvVar:   "OPENAI_API_KEY",
			Required: true,
		},
		cli.StringFlag{
			Name:     "ponyuser",
			Usage:    "SSH pony user for authentication",
			EnvVar:   "PONY_SSH_USER",
			Required: true,
		},
		cli.StringFlag{
			Name:   "ponypass",
			Usage:  "SSH pony password for authentication",
			EnvVar: "PONY_SSH_PASS",
			Value:  "",
		},
	}
	err := app.Run(os.Args)
	if err != nil {
		log.Fatal(err)
	}
}

func sshHandler(s ssh.Session, ponyUser string, openaiApiKey string) {
	if s.User() != ponyUser {
		_ = s.Exit(1)
		return
	}
	pty, winCh, isPty := s.Pty()
	if !isPty {
		_, _ = io.WriteString(s, "No PTY requested.\n")
		s.Close()
		return
	}
	process := newProcessor(openaiApiKey, s, pty.Window.Width)

	// display the pony
	pony, err := process.getPony()
	if err != nil {
		log.Printf("failed to deliver pony: %s", err)
		s.Close()
		return
	}
	if _, err := s.Write(pony); err != nil {
		log.Printf("failed to write pony: %s", err)
		s.Close()
		return
	}
	_, _ = io.WriteString(s, "\n")

	// set up the terminal
	term := term.NewTerminal(s, " > ")
	_ = term.SetSize(pty.Window.Width, pty.Window.Height)

	// consume the window size changes
	// it is important to do so to prevent a deadlock on that channel
	go func() {
		for win := range winCh {
			_ = term.SetSize(win.Width, win.Height)
			process.setWidth(win.Width)
		}
	}()

	// conversation loop
	for {
		line, err := term.ReadLine()
		if err != nil {
			// err == io.EOF when user presses ctrl+d
			if !errors.Is(err, io.EOF) {
				fmt.Println(err)
			}
			s.Close()
			return
		}
		if line == "quit" {
			s.Close()
			return
		}
		if err := process.processUserLine(s.Context(), line); err != nil {
			log.Printf("failed to process user line: %s", err)
			_, _ = io.WriteString(s, "Sorry, I don't know what to say. Bye.\n")
			_ = s.Exit(2)
			return
		}
	}
}

func serve(c *cli.Context) error {
	hostkey, err := os.ReadFile(c.GlobalString("hostkey"))
	if err != nil {
		return err
	}
	requiredPass := c.GlobalString("ponypass")

	var wg sync.WaitGroup

	ssh.Handle(func(s ssh.Session) {
		sshHandler(s, c.GlobalString("ponyuser"), c.GlobalString("openai-api-key"))
	})

	wg.Add(1)
	go func() {
		opts := []ssh.Option{
			ssh.HostKeyPEM(hostkey),
		}
		if requiredPass != "" {
			opts = append(opts, ssh.PasswordAuth(func(ctx ssh.Context, pass string) bool {
				return pass == requiredPass
			}))
		}
		if err := ssh.ListenAndServe(c.GlobalString("ssh-addr"), nil, opts...); err != nil {
			log.Println(err)
		}
		wg.Done()
	}()
	wg.Wait()

	return nil
}

type processor struct {
	openaiApiKey         string
	session              ssh.Session
	ponyCodeName         string
	ponyName             string
	client               *openai.Client
	previousMesssages    []openai.ChatCompletionMessage
	maxPreviousMesssages int
	textReponseFormat    openai.ChatCompletionResponseFormat
	width                *int32
}

func newProcessor(openaiApiKey string, session ssh.Session, initialWidth int) *processor {
	ponyCodeName := listOfPonies[rand.Intn(len(listOfPonies))]
	caser := cases.Title(language.English)
	ponyName := caser.String(ponyCodeName)
	client := openai.NewClient(openaiApiKey)
	message := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: fmt.Sprintf(PROMPT_TEMPLATE, ponyName),
	}
	proc := &processor{
		openaiApiKey:         openaiApiKey,
		session:              session,
		ponyCodeName:         ponyCodeName,
		ponyName:             ponyName,
		client:               client,
		textReponseFormat:    openai.ChatCompletionResponseFormat{Type: openai.ChatCompletionResponseFormatTypeText},
		maxPreviousMesssages: 20,
		width:                new(int32),
	}
	*proc.width = int32(initialWidth)
	proc.addMessage(message)
	return proc
}

func (p *processor) getWidth() int {
	return int(atomic.LoadInt32(p.width))
}

func (p *processor) setWidth(w int) {
	atomic.StoreInt32(p.width, int32(w))
}

func (p *processor) addMessage(msg openai.ChatCompletionMessage) {
	p.previousMesssages = append(p.previousMesssages, msg)
	if len(p.previousMesssages) > p.maxPreviousMesssages {
		// keep the initial prompt
		newMessages := p.previousMesssages[0:1]
		// take away the oldest message
		newMessages = append(newMessages, p.previousMesssages[2:]...)
		p.previousMesssages = newMessages
	}
}

func (p *processor) processUserLine(ctx context.Context, line string) error {
	userMessage := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: line,
	}
	p.addMessage(userMessage)
	req := openai.ChatCompletionRequest{
		Model:          openai.GPT3Dot5Turbo1106,
		Messages:       p.previousMesssages,
		ResponseFormat: &p.textReponseFormat,
	}
	resp, err := p.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to create chat completion: %w", err)
	}
	first := resp.Choices[0]
	if first.FinishReason != openai.FinishReasonStop {
		return fmt.Errorf("unexpected finish reason: %s", first.FinishReason)
	}
	if err := p.writeResponse(first); err != nil {
		return fmt.Errorf("failed to write response: %w", err)
	}

	p.addMessage(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: first.Message.Content,
	})

	return nil
}

func (p *processor) writeResponse(choice openai.ChatCompletionChoice) error {
	respContent := strings.TrimSpace(choice.Message.Content)
	respContentWrapped := wordwrap.String(respContent, p.getWidth()-4)
	lines := strings.Split(respContentWrapped, "\n")
	var b bytes.Buffer
	b.WriteString("\n")
	for _, line := range lines {
		b.WriteString(" ")
		b.WriteString(line)
		b.WriteString("\n")
	}
	b.WriteString("\n")
	if _, err := b.WriteTo(p.session); err != nil {
		return fmt.Errorf("failed to write response: %w", err)
	}
	return nil
}

func (p *processor) getPony() ([]byte, error) {
	say := fmt.Sprintf("Hello! My name is %s!", p.ponyName)
	cmd := exec.Command("ponysay", "-W", strconv.Itoa(p.getWidth()), "-X", "-b", "ascii", "--pony", p.ponyCodeName, say)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	if err := cmd.Run(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
