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

	"github.com/gliderlabs/ssh"
	"github.com/joho/godotenv"
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
		return
	}
	pty, winCh, isPty := s.Pty()
	if !isPty {
		io.WriteString(s, "No PTY requested.\n")
		return
	}
	process := newProcessor(openaiApiKey)

	// display the pony
	pony, err := process.getPony(pty.Window.Width)
	if err != nil {
		log.Printf("failed to deliver pony: %s", err)
		return
	}
	s.Write(pony)
	io.WriteString(s, "\n")

	// set up the terminal
	term := term.NewTerminal(s, "> ")
	term.SetSize(pty.Window.Width, pty.Window.Height)

	// conversation loop
	for {
		line, err := term.ReadLine()
		if err != nil {
			// err == io.EOF when user presses ctrl+d
			if !errors.Is(err, io.EOF) {
				fmt.Println(err)
			}
			break
		}
		if line == "quit" {
			break
		}
		if err := process.processUserLine(context.Background(), s, line); err != nil {
			log.Printf("failed to process user line: %s", err)
			io.WriteString(s, "Sorry, I don't know what to say. Bye.\n")
			break
		}
		io.WriteString(s, "\n\n")
	}

	go func() {
		for win := range winCh {
			term.SetSize(win.Height, win.Width)
		}
	}()
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
	ponyCodeName         string
	ponyName             string
	client               *openai.Client
	previousMesssages    []openai.ChatCompletionMessage
	maxPreviousMesssages int
	textReponseFormat    openai.ChatCompletionResponseFormat
}

func newProcessor(openaiApiKey string) *processor {
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
		ponyCodeName:         ponyCodeName,
		ponyName:             ponyName,
		client:               client,
		textReponseFormat:    openai.ChatCompletionResponseFormat{Type: openai.ChatCompletionResponseFormatTypeText},
		maxPreviousMesssages: 20,
	}
	proc.addMessage(message)
	return proc
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

func (p *processor) processUserLine(ctx context.Context, s ssh.Session, line string) error {
	userMessage := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: line,
	}
	p.addMessage(userMessage)
	request := openai.ChatCompletionRequest{
		Model:          openai.GPT3Dot5Turbo1106,
		Messages:       p.previousMesssages,
		ResponseFormat: p.textReponseFormat,
	}
	stream, err := p.client.CreateChatCompletionStream(ctx, request)
	if err != nil {
		return fmt.Errorf("failed to create chat completion stream: %w", err)
	}
	defer stream.Close()
	var fullResponse strings.Builder
	for {
		response, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to receive chat completion stream: %w", err)
		}
		delta := response.Choices[0].Delta.Content
		fullResponse.WriteString(delta)
		io.WriteString(s, delta)
	}
	p.addMessage(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: fullResponse.String(),
	})
	return nil

}

func (p *processor) getPony(width int) ([]byte, error) {
	say := fmt.Sprintf("Hello! My name is %s!", p.ponyName)
	cmd := exec.Command("ponysay", "-W", strconv.Itoa(width), "-X", "-b", "ascii", "--pony", p.ponyCodeName, say)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	if err := cmd.Run(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
