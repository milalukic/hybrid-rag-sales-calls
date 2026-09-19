package com.milalukic.raggateway.controller;

import com.milalukic.raggateway.dto.QueryRequest;
import com.milalukic.raggateway.dto.QueryResponse;
import com.milalukic.raggateway.service.RagQueryService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class QueryController {

    private final RagQueryService ragQueryService;

    public QueryController(RagQueryService ragQueryService) {
        this.ragQueryService = ragQueryService;
    }

    @PostMapping("/query")
    public ResponseEntity<QueryResponse> query(@Valid @RequestBody QueryRequest request) {
        return ResponseEntity.ok(ragQueryService.query(request));
    }

    // Open endpoint (not behind the API key check) so deploy platforms can health-check it.
    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of("status", "ok"));
    }
}
