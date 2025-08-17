FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install application
COPY . /app
RUN pip install --no-cache-dir .

# Default environment variables
ENV FLASK_HOST=0.0.0.0 \
    FLASK_PORT=3000 \
    MILVUS_HOST=milvus \
    MILVUS_PORT=19530

# Expose port
EXPOSE 3000

# Run the application
CMD ["python", "app.py"]
