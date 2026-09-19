package com.milalukic.raggateway.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Positive;

// sta android salje gateway-u, i sta gateway salje api-u.
// QueryRequest = isti kao u python kodu

public class QueryRequest {

    @NotBlank(message = "question must not be blank")
    private String question;

    // Optional metadata filters, same as hybrid_search(stage=..., customer=...)
    private String stage;
    private String customer;

    @Positive
    @JsonProperty("top_k") // u pythonu je top_k, ovde koristi topK
    private Integer topK = 3;

    public QueryRequest() {
    }

    public String getQuestion() {
        return question;
    }

    public void setQuestion(String question) {
        this.question = question;
    }

    public String getStage() {
        return stage;
    }

    public void setStage(String stage) {
        this.stage = stage;
    }

    public String getCustomer() {
        return customer;
    }

    public void setCustomer(String customer) {
        this.customer = customer;
    }

    public Integer getTopK() {
        return topK;
    }

    public void setTopK(Integer topK) {
        this.topK = topK;
    }

    public String cacheKey() {
        return String.join("|",
                safe(question).trim().toLowerCase(),
                safe(stage),
                safe(customer),
                String.valueOf(topK == null ? 3 : topK));
    }

    private static String safe(String value) {
        return value == null ? "" : value;
    }
}
