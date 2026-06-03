FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV / PIL
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create dirs for uploads and reports
RUN mkdir -p /app/uploads /app/reports

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]