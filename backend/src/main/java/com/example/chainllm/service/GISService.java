package com.example.chainllm.service;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.Map;

@Service
@RequiredArgsConstructor
public class GISService {

    private final WebClient.Builder webClientBuilder;

    public void sendLlmResultToPython(String taskId, String prompt, String llmResponse) {
        String pythonUrl = "http://localhost:8000/llm/result";

        Map<String, String> payload = Map.of(
                "taskId", taskId,
                "prompt", prompt,
                "response", llmResponse
        );

        webClientBuilder.build()
                .post()
                .uri(pythonUrl)
                .bodyValue(payload)
                .retrieve()
                .bodyToMono(Void.class)
                .onErrorResume(e -> {
                    System.err.println("Failed to send LLM result to Python: " + e.getMessage());
                    return Mono.empty();
                })
                .subscribe();
    }
}
