package com.example.chainllm.controller;

import com.example.chainllm.dto.LLMRequest;
import com.example.chainllm.dto.LLMResponse;
import com.example.chainllm.service.LLMService;
import com.example.chainllm.service.GISService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/llm")
@RequiredArgsConstructor
public class LLMController {

    private final LLMService llmService;
    private final GISService gisService;

    @PostMapping("/ask")
    public String askLlm(@RequestBody LLMRequest request) {
        // 1. Generate and return taskId
        String taskId = UUID.randomUUID().toString();

        // 2. Process in background
        new Thread(() -> {
            try {
                // a. Get LLM response from Ollama
                LLMResponse llmResponse = llmService.getLLMResponse(request);

                // b. Send to GIS server with taskId
                gisService.sendLlmResultToPython(taskId, request.getPrompt(), llmResponse.getResponse());

            } catch (Exception e) {
                e.printStackTrace(); // or log properly
            }
        }).start();

        return taskId; // Immediate return to frontend
    }
}
