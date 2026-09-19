package com.milalukic.raggateway.config;

import com.github.benmanes.caffeine.cache.Caffeine;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.CacheManager;
import org.springframework.cache.caffeine.CaffeineCacheManager;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.TimeUnit;

@Configuration
public class CacheConfig {

    public static final String QUERY_CACHE = "ragQueryCache";

    @Bean
    public CacheManager cacheManager(
            @Value("${rag.cache.ttl-seconds}") long ttlSeconds,
            @Value("${rag.cache.max-size}") long maxSize
    ) {
        CaffeineCacheManager manager = new CaffeineCacheManager(QUERY_CACHE);
        manager.setCaffeine(Caffeine.newBuilder()
                .expireAfterWrite(ttlSeconds, TimeUnit.SECONDS)
                .maximumSize(maxSize));
        return manager;
    }
}
