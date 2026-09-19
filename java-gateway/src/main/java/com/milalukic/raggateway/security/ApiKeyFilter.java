package com.milalukic.raggateway.security;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

// filter - kod koji runnuje na svakom http requestu pre nego sto dodje do kontrolera
    // ako x-api-key nije isti kao tajna koju imamo onda se vraca 401

@Component
public class ApiKeyFilter extends OncePerRequestFilter {

    private final String expectedApiKey;

    public ApiKeyFilter(@Value("${rag.gateway.api-key}") String expectedApiKey) {
        this.expectedApiKey = expectedApiKey;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                     HttpServletResponse response,
                                     FilterChain filterChain) throws ServletException, IOException {

        String path = request.getRequestURI();

        // Health check stays open so deploy platforms can ping it.
        if (!path.startsWith("/api/") || path.equals("/api/health")) {
            filterChain.doFilter(request, response);
            return;
        }

        String providedKey = request.getHeader("X-API-Key");

        if (providedKey == null || !providedKey.equals(expectedApiKey)) {
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.setContentType("application/json");
            response.getWriter().write("{\"error\":\"missing or invalid X-API-Key header\"}");
            return;
        }

        filterChain.doFilter(request, response);
    }
}
