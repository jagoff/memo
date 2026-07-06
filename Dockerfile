FROM python:3.14-slim

WORKDIR /app

# Install uv + git (needed for dependencies)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy repo
COPY . .

# Install dependencies via uv
RUN uv sync

# Set environment
ENV PYTHONUNBUFFERED=1

# Run memo-mcp server
CMD ["memo-mcp"]
