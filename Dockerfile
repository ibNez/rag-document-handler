FROM python:alpine3.21

# Set work directory
WORKDIR /app

# Install system dependencies for production
RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    curl \
    && rm -rf /var/cache/apk/*

# Install application
COPY . /app
RUN pip install --no-cache-dir . gunicorn

# Create non-root user for security
RUN addgroup -g 1001 -S appgroup && \
    adduser -S -D -H -u 1001 -h /app -s /sbin/nologin -G appgroup appuser

# Create directories and set ownership
RUN mkdir -p /app/ssl/certs /app/ssl/private /app/ssl/config /app/logs && \
    chown -R appuser:appgroup /app && \
    chmod +x /app/start_production.sh /app/stop_production.sh /app/status_production.sh && \
    chmod 755 /app/ssl && \
    chmod 700 /app/ssl/private

# Default environment variables for production
ENV FLASK_ENV=production \
    FLASK_DEBUG=False \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=443 \
    WORKERS=4 \
    WORKER_CLASS=sync \
    BIND_ADDRESS=0.0.0.0:443 \
    MAX_REQUESTS=1000 \
    TIMEOUT=120 \
    MILVUS_HOST=milvus \
    MILVUS_PORT=19530

# Expose HTTPS port
EXPOSE 443

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f -k https://localhost:443/health || exit 1

# Run the application in production mode
CMD ["/app/start_production.sh"]
