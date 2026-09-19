package com.milalukic.raggateway.service;

import com.milalukic.raggateway.config.CacheConfig;
import com.milalukic.raggateway.dto.QueryRequest;
import com.milalukic.raggateway.dto.QueryResponse;
import org.springframework.cache.Cache;
import org.springframework.cache.CacheManager;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

// proveri cache prvo pa ako je miss idi na source i onda populateujemo cache sledeci put

@Service
public class RagQueryService {

    private final WebClient pythonServiceWebClient;
    private final CacheManager cacheManager;

    public RagQueryService(WebClient pythonServiceWebClient, CacheManager cacheManager) {
        this.pythonServiceWebClient = pythonServiceWebClient;
        this.cacheManager = cacheManager;
    }

    public QueryResponse query(QueryRequest request) {
        Cache cache = cacheManager.getCache(CacheConfig.QUERY_CACHE);
        String key = request.cacheKey();

        if (cache != null) {
            QueryResponse cached = cache.get(key, QueryResponse.class);
            if (cached != null) {
                return cached.asCached();
            }
        }

        QueryResponse fresh = callPythonService(request);

        if (cache != null) {
            cache.put(key, fresh);
        }

        return fresh;
    }

    private QueryResponse callPythonService(QueryRequest request) {
        try {
            return pythonServiceWebClient.post()
                    .uri("/query")
                    .bodyValue(request)
                    .retrieve()
                    .bodyToMono(QueryResponse.class)
                    .block();
        } catch (WebClientResponseException e) {
            throw new UpstreamServiceException(
                    "Python retrieval service returned " + e.getStatusCode(), e);
        } catch (Exception e) {
            throw new UpstreamServiceException(
                    "Could not reach Python retrieval service", e);
        }
    }
}
