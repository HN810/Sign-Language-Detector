FROM python:3.11-slim

WORKDIR /app

# Copy only the frontend server and requirements (model should be provided separately or baked into image)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000

# Bind to all interfaces so hosting platforms can route traffic in. The server will
# respect the PORT environment variable if provided.
ENV HOST=0.0.0.0
ENV PORT=5000

CMD ["python", "server.py"]
