FROM python:3.10-slim

# Install system dependencies (FFmpeg for media processing, libgl for OpenCV)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]