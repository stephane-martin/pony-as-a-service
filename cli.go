package main

import "github.com/urfave/cli"

func makeApp(action cli.ActionFunc) *cli.App {
	app := cli.NewApp()
	app.Name = "pony-as-a-service"
	app.Usage = "deliver ponies as a service"
	app.Action = action
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
			Name:   "openai-api-key",
			Usage:  "OpenAI API key",
			EnvVar: "OPENAI_API_KEY",
			Value:  "",
		},
		cli.StringFlag{
			Name:   "azure-openai-access-key",
			Usage:  "Azure OpenAI access key",
			EnvVar: "AZURE_OPENAI_ACCESS_KEY",
			Value:  "",
		},
		cli.StringFlag{
			Name:   "azure-openai-endpoint",
			Usage:  "When using Azure OpenAI, the endpoint to use",
			EnvVar: "AZURE_OPENAI_ENDPOINT",
			Value:  "",
		},
		cli.StringFlag{
			Name:   "azure-openai-deployment-name",
			Usage:  "When using Azure OpenAI, the deployment name for gpt-35-turbo-16k",
			EnvVar: "AZURE_OPENAI_DEPLOYMENT_NAME",
			Value:  "",
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
		cli.BoolFlag{
			Name:   "syslog",
			Usage:  "Log to syslog",
			EnvVar: "PONY_SYSLOG",
		},
	}
	return app
}
