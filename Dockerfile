# Use the official TensorFlow image as base
FROM python:3.9-slim

RUN pip3 install tensorflow==2.15.0

# Set working directory
WORKDIR /app

RUN echo "Acquire::Check-Valid-Until \"false\";" > /etc/apt/apt.conf.d/10no-check-valid-until && \
    echo "Acquire::Check-Date \"false\";" > /etc/apt/apt.conf.d/10no-check-date

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Create directory for model storage
RUN mkdir -p /app/model

# Copy static files first
COPY static /app/static

# Copy the rest of the application code
COPY . /app/

# Expose ports
EXPOSE 80 5000

# Run the application
CMD ["python3", "app.py"]
