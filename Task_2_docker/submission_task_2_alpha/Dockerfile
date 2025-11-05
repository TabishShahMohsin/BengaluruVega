FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY codescripts/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY codescripts/ .

# Ensure ref_data folder exists
RUN mkdir -p /app/ref_data

# Copy SDR, speed parquet, and GraphML into container
COPY ref_data/processed_hex_avg_SDR.parquet /app/ref_data/processed_hex_avg_SDR.parquet
COPY ref_data/smoothed_speed_full.parquet /app/ref_data/smoothed_speed_full.parquet


# Ensure output directory exists
RUN mkdir -p /app/out /app/data

# Entrypoint and default CMD
ENTRYPOINT ["python", "/app/main.py"]
CMD ["--input-csv", "/app/data/input.csv", "--output-json", "/app/out/output.json"]
