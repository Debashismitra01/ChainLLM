package com.example.chainllm.service;

import com.example.chainllm.dto.LLMRequest;
import com.example.chainllm.dto.LLMResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
@RequiredArgsConstructor
public class LLMService {

    private final WebClient.Builder webClientBuilder;

    public LLMResponse getLLMResponse(LLMRequest request) {

        String ollamaUrl = "http://localhost:11434/api/generate"; // Replace if remote

        String result = webClientBuilder.build()
                .post()
                .uri(ollamaUrl)
                .bodyValue(new OllamaGenerateRequest("mistral", request.getPrompt()))
                .retrieve()
                .bodyToMono(OllamaGenerateResponse.class)
                .block()
                .response();

        return new LLMResponse(result);
    }

    // inner DTOs for Ollama JSON
    record OllamaGenerateRequest(String model, String prompt) {}

    record OllamaGenerateResponse(String response) {}
}
