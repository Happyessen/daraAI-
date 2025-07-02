FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt
RUN pip install gunicorn

# Copy application
COPY . .

# Create directories
RUN mkdir -p uploads static templates

# Expose port
EXPOSE 8000

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "app:app"]