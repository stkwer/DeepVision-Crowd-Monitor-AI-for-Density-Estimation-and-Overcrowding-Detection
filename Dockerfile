# Step 1 — Base Image
FROM python:3.10-slim

# Step 2 — Prevent interaction
ENV DEBIAN_FRONTEND=noninteractive

# Step 3 — Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 4 — Create app directory
WORKDIR /app

# Step 5 — Copy project files
COPY . /app

# Step 6 — Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 7 — Expose Streamlit port
EXPOSE 8501

# Step 8 — Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
