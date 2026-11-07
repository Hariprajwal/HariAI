# HariAI Minimal Container Image
FROM python:3.11-slim

LABEL maintainer="K R Hari Prajwal <hariprajwal77@gmail.com>"
LABEL description="HariAI: Compact, open-weight AI system under 50 MB"
LABEL org.opencontainers.image.source="https://github.com/Hariprajwal"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY hari_ai/ ./hari_ai/
COPY tests/ ./tests/
COPY plan.md .

# Install package
RUN pip install --no-cache-dir .

# Default entrypoint runs HariAI CLI
ENTRYPOINT ["hari"]
CMD ["run"]

