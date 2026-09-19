package com.milalukic.raggateway.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * One retrieved chunk, matching hybrid-rag-sales-calls' RetrievedChunk
 * (Pydantic model in api.py) exactly: {text, customer, deal_stage, score}.
 */
public record RetrievedChunk(
        String text,
        String customer,
        @JsonProperty("deal_stage") String dealStage,
        double score
) {
}
