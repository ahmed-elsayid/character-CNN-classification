# Use Python 3.11 as specified in your setup instructions
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt /app/

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app/

# Expose the default Hugging Face Space port
EXPOSE 7860

# Command to run the application
CMD["python", "app.py"]