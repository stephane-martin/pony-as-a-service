package main

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

var TextReponseFormat = &openai.ChatCompletionResponseFormat{Type: openai.ChatCompletionResponseFormatTypeText}

type Processor interface {
	AddMessage(msg openai.ChatCompletionMessage)
	GetClient() *openai.Client
	WriteResponse(response string) error
	GetPreviousMessages() []openai.ChatCompletionMessage
}

func ProcessUserLine(ctx context.Context, p Processor, line string) error {
	userMessage := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: line,
	}
	p.AddMessage(userMessage)
	req := openai.ChatCompletionRequest{
		Model:          openai.GPT3Dot5Turbo1106,
		Messages:       p.GetPreviousMessages(),
		ResponseFormat: TextReponseFormat,
	}
	resp, err := p.GetClient().CreateChatCompletion(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to create chat completion: %w", err)
	}
	first := resp.Choices[0]
	if first.FinishReason != openai.FinishReasonStop {
		return fmt.Errorf("unexpected finish reason: %s", first.FinishReason)
	}
	if err := p.WriteResponse(first.Message.Content); err != nil {
		return fmt.Errorf("failed to write response: %w", err)
	}

	p.AddMessage(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: first.Message.Content,
	})

	return nil
}
