package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"log/syslog"
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
	"time"

	"github.com/gliderlabs/ssh"
	"github.com/hashicorp/go-retryablehttp"
	"github.com/joho/godotenv"
	"github.com/muesli/reflow/wordwrap"
	"github.com/sashabaranov/go-openai"
	slogsysloghandler "github.com/stephane-martin/slog-syslog-handler"
	"github.com/urfave/cli"
	"golang.org/x/term"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

const MODEL = openai.GPT3Dot5Turbo16K

var logger = slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

func main() {
	_ = godotenv.Load()
	app := makeApp(serve)
	if err := app.Run(os.Args); err != nil {
		logger.Error(err.Error())
		os.Exit(1)
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
	retryClient := retryablehttp.NewClient()
	retryClient.RetryMax = 3
	retryClient.Logger = logger
	retryClient.HTTPClient.Timeout = 10 * time.Minute
	nativeClient := retryClient.StandardClient()
	if d.openAIApiKey != "" {
		config := openai.DefaultConfig(d.openAIApiKey)
		config.HTTPClient = nativeClient
		return openai.NewClientWithConfig(config)
	}
	config := openai.DefaultAzureConfig(d.azureOpenAIAccessKey, d.azureOpenAIEndpoint)
	config.HTTPClient = nativeClient
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
	if c.GlobalBool("syslog") {
		logger.Info("connecting to syslog")
		syslogger, err := syslog.New(syslog.LOG_INFO|syslog.LOG_LOCAL3, "ponies")
		if err != nil {
			return fmt.Errorf("failed to connect to syslog: %w", err)
		}
		logger.Info("connected to syslog: stop writing logs on stderr")
		logger = slogsysloghandler.NewLogger(syslogger, true, &slog.HandlerOptions{Level: slog.LevelInfo})
	}
	// read the SSH host key
	hostkey, err := os.ReadFile(c.GlobalString("hostkey"))
	if err != nil {
		return fmt.Errorf("failed to read SSH host key: %w", err)
	}

	// read the deployment options from the environment / command line
	dopts, err := newDeploymentOptions(c)
	if err != nil {
		return err
	}

	// listen on the SSH port
	sshListenAddr := c.GlobalString("ssh-addr")
	sshListener, err := net.Listen("tcp", sshListenAddr)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", sshListenAddr, err)
	}
	defer sshListener.Close()
	logger.Info("SSH listener", "addr", sshListenAddr)

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
		logger.Info("ssh server starting")
		if err := sshServer.Serve(sshListener); err != nil {
			logger.Error("ssh service stopped", "error", err)
		}
		wg.Done()
	}()
	wg.Wait()

	return nil
}

func sshHandler(s ssh.Session, ponyUser string, opts *deploymentOptions) {
	logger.Info("new SSH connection", "user", s.User(), "remote_addr", s.RemoteAddr())
	if s.User() != ponyUser {
		logger.Warn("user not allowed to log in", "user", s.User())
		_ = s.Exit(1)
		return
	}
	pty, winCh, isPty := s.Pty()
	if !isPty {
		_, _ = io.WriteString(s, "No PTY requested.\n")
		logger.Info("no PTY requested")
		s.Close()
		return
	}
	process := newSSHProcessor(opts, s, pty.Window.Width)

	// display the pony
	if err := process.writePony(); err != nil {
		logger.Warn("failed to deliver pony", "error", err)
		s.Close()
		return
	}

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
				logger.Warn("failed to read user line", "error", err)
			}
			s.Close()
			return
		}
		if line == "quit" {
			s.Close()
			return
		}
		if err := ProcessUserLine(s.Context(), process, line); err != nil {
			logger.Warn("failed to process user line", "error", err)
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

func (p *sshProcessor) writePony() error {
	say := fmt.Sprintf("Hello! My name is %s!", p.ponyName)
	cmd := exec.Command("ponysay", "-W", strconv.Itoa(p.getWidth()), "-X", "-b", "ascii", "--pony", p.ponyCodeName, say)
	cmd.Stdout = p.session
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to run ponysay: %w", err)
	}
	return nil
}
