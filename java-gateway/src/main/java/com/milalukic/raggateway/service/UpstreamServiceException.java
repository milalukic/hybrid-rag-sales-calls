package com.milalukic.raggateway.service;

public class UpstreamServiceException extends RuntimeException {
    public UpstreamServiceException(String message, Throwable cause) {
        super(message, cause);
    }
}
