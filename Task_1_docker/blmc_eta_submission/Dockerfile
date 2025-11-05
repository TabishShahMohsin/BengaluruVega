# Base image
FROM python:3.11-slim

# Install system dependencies (needed for geopy, pyarrow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

# Copy your scripts
COPY codescripts/ ./codescripts/

# Copy reference data into /app/refdata (will not be overwritten by mounts)
COPY refdata/ /app/refdata/
# Create /app/data (empty, used for inputs/outputs provided by testers)
RUN mkdir -p /app/data
RUN mkdir -p /app/out

# Debug (optional - can remove later)
RUN ls -R /app

# Entrypoint
ENTRYPOINT ["python", "/app/codescripts/prediction.py"]
CMD ["--input-json", "/app/data/input.json", "--output-json", "/app/out/output.json"]

