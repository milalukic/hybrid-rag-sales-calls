// Ovo je kao power switch koji SpringBoothApp-u kaze da skenira paket, poveze komponente, zapocne embedded
// veb server. enable caching obavezan da bi cachable radio

package com.milalukic.raggateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;

@SpringBootApplication
@EnableCaching
public class RagGatewayApplication {
    public static void main(String[] args) {
        SpringApplication.run(RagGatewayApplication.class, args);
    }
}
