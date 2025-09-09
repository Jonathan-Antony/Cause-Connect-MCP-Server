# Use lightweight Python base image
FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user early
RUN useradd -m -u 1001 mcpserver

# Set working directory
WORKDIR /app

# Copy Python dependency files (if you have requirements.txt or pyproject.toml)
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . .

# Change ownership of /app to mcpserver user
RUN chown -R mcpserver:mcpserver /app

# Switch to non-root user AFTER setting permissions
USER mcpserver

# Expose the port (matches your FastAPI main.py default)
EXPOSE 8003

# Health check – call the health endpoint you added
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8003/health || exit 1

# Start the server
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8003"]