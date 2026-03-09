# Dockerfile
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project code
COPY . .

# Expose port
EXPOSE 8000

# Run migrations and start server (for local testing)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]