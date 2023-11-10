package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"

	"github.com/gliderlabs/ssh"
	"github.com/joho/godotenv"
	"github.com/muesli/reflow/wordwrap"
	"github.com/sashabaranov/go-openai"
	"github.com/urfave/cli"
	"golang.org/x/term"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

const MODEL = openai.GPT3Dot5Turbo16K

func main() {
	_ = godotenv.Load()
	app := makeApp()
	if err := app.Run(os.Args); err != nil {
		log.Fatal(err)
	}
}

type deploymentOptions struct {
	openAIApiKey              string
	azureOpenAIAccessKey      string
	azureOpenAIEndpoint       string
	azureOpenAIDeploymentName string
}

func (opts deploymentOptions) check() error {
	if opts.openAIApiKey == "" && opts.azureOpenAIAccessKey == "" {
		return errors.New("either OpenAI or Azure API key must be provided")
	}
	if opts.openAIApiKey != "" && opts.azureOpenAIAccessKey != "" {
		return errors.New("only one of OpenAI or Azure API key must be provided")
	}
	if opts.azureOpenAIAccessKey != "" && opts.azureOpenAIDeploymentName == "" {
		return errors.New("azure endpoint must be provided")
	}
	return nil
}

func newDeploymentOptions(c *cli.Context) (*deploymentOptions, error) {
	opts := deploymentOptions{
		openAIApiKey:              c.GlobalString("openai-api-key"),
		azureOpenAIAccessKey:      c.GlobalString("azure-openai-access-key"),
		azureOpenAIEndpoint:       c.GlobalString("azure-openai-endpoint"),
		azureOpenAIDeploymentName: c.GlobalString("azure-openai-deployment-name"),
	}
	if err := opts.check(); err != nil {
		return nil, err
	}
	return &opts, nil
}

func makeOpenAIClient(d *deploymentOptions) *openai.Client {
	if d.openAIApiKey != "" {
		return openai.NewClient(d.openAIApiKey)
	}
	config := openai.DefaultAzureConfig(d.azureOpenAIAccessKey, d.azureOpenAIEndpoint)
	if d.azureOpenAIDeploymentName != "" {
		config.AzureModelMapperFunc = func(model string) string {
			azureModelMapping := map[string]string{
				MODEL: d.azureOpenAIDeploymentName,
			}
			return azureModelMapping[model]
		}
	}
	return openai.NewClientWithConfig(config)
}

func serve(c *cli.Context) error {
	hostkey, err := os.ReadFile(c.GlobalString("hostkey"))
	if err != nil {
		return fmt.Errorf("failed to read SSH host key: %w", err)
	}
	dopts, err := newDeploymentOptions(c)
	if err != nil {
		return err
	}
	// capture signals
	sigchan := make(chan os.Signal, 1)
	signal.Notify(sigchan, os.Interrupt, syscall.SIGTERM)

	// create a global context to manage the lifetime of all the services
	ctx, cancel := context.WithCancel(context.Background())

	// when a signal is received, cancel the context
	go func() {
		<-sigchan
		cancel()
	}()

	requiredPass := c.GlobalString("ponypass")

	sshListenAddr := c.GlobalString("ssh-addr")
	sshListener, err := net.Listen("tcp", sshListenAddr)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", sshListenAddr, err)
	}
	defer sshListener.Close()

	opts := []ssh.Option{
		ssh.HostKeyPEM(hostkey),
	}
	if requiredPass != "" {
		opts = append(opts, ssh.PasswordAuth(func(ctx ssh.Context, pass string) bool {
			return pass == requiredPass
		}))
	}
	// the SSH handler to be called when a new SSH connection is opened
	h := func(s ssh.Session) {
		sshHandler(s, c.GlobalString("ponyuser"), dopts)
	}

	sshServer := &ssh.Server{Handler: h}
	for _, option := range opts {
		if err := sshServer.SetOption(option); err != nil {
			return err
		}
	}

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		<-ctx.Done()
		sshServer.Close()
		wg.Done()
	}()

	wg.Add(1)
	go func() {
		if err := sshServer.Serve(sshListener); err != nil {
			log.Println(err)
		}
		wg.Done()
	}()
	wg.Wait()

	return nil
}

func sshHandler(s ssh.Session, ponyUser string, opts *deploymentOptions) {
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
	process := newSSHProcessor(opts, s, pty.Window.Width)

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
		if err := ProcessUserLine(s.Context(), process, line); err != nil {
			log.Printf("failed to process user line: %s", err)
			_, _ = io.WriteString(s, "Sorry, I don't know what to say. Bye.\n")
			_ = s.Exit(2)
			return
		}
	}
}

type sshProcessor struct {
	session              ssh.Session
	ponyCodeName         string
	ponyName             string
	client               *openai.Client
	previousMesssages    []openai.ChatCompletionMessage
	maxPreviousMesssages int
	width                *int32
}

func newSSHProcessor(opts *deploymentOptions, session ssh.Session, initialWidth int) *sshProcessor {
	ponyCodeName := listOfPonies[rand.Intn(len(listOfPonies))]
	caser := cases.Title(language.English)
	ponyName := caser.String(ponyCodeName)
	client := makeOpenAIClient(opts)
	initMessage := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: fmt.Sprintf(PROMPT_TEMPLATE, ponyName),
	}
	proc := &sshProcessor{
		session:              session,
		ponyCodeName:         ponyCodeName,
		ponyName:             ponyName,
		client:               client,
		maxPreviousMesssages: 20,
		width:                new(int32),
	}
	*proc.width = int32(initialWidth)
	proc.AddMessage(initMessage)
	return proc
}

func (p *sshProcessor) GetClient() *openai.Client {
	return p.client
}

func (p *sshProcessor) GetPreviousMessages() []openai.ChatCompletionMessage {
	return p.previousMesssages
}

func (p *sshProcessor) getWidth() int {
	return int(atomic.LoadInt32(p.width))
}

func (p *sshProcessor) setWidth(w int) {
	atomic.StoreInt32(p.width, int32(w))
}

func (p *sshProcessor) AddMessage(msg openai.ChatCompletionMessage) {
	p.previousMesssages = append(p.previousMesssages, msg)
	if len(p.previousMesssages) > p.maxPreviousMesssages {
		// keep the initial prompt
		newMessages := p.previousMesssages[0:1]
		// take away the oldest message
		newMessages = append(newMessages, p.previousMesssages[2:]...)
		p.previousMesssages = newMessages
	}
}

func (p *sshProcessor) WriteResponse(resp string) error {
	respWrapped := wordwrap.String(resp, p.getWidth()-4)
	lines := strings.Split(respWrapped, "\n")
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

func (p *sshProcessor) getPony() ([]byte, error) {
	say := fmt.Sprintf("Hello! My name is %s!", p.ponyName)
	cmd := exec.Command("ponysay", "-W", strconv.Itoa(p.getWidth()), "-X", "-b", "ascii", "--pony", p.ponyCodeName, say)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	if err := cmd.Run(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
