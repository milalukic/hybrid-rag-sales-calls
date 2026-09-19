package com.milalukic.raggateway.dto;

import java.util.List;

public record QueryResponse(
        String answer,
        List<RetrievedChunk> retrieved,
        boolean fromCache
) {
    public static QueryResponse fromUpstream(String answer, List<RetrievedChunk> retrieved) {
        return new QueryResponse(answer, retrieved, false);
    }

    public QueryResponse asCached() {
        return new QueryResponse(this.answer, this.retrieved, true);
    }
}
