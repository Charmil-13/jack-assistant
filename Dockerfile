# Use the official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8502
ENV PORT=8502

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Expose port 80 (Azure requires the container to listen on port 80)
EXPOSE 8502

# Run Streamlit with configuration suitable for Azure App Service
CMD ["streamlit", "run", "jack.py", "--server.port=8502", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
