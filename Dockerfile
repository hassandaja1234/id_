# Use Python 3.11.0 slim as base image
FROM python:3.11.0-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (Render uses PORT environment variable)
EXPOSE 8000

# Command to run the application
CMD uvicorn card_reader_api:app --host 0.0.0.0 --port ${PORT:-8000}