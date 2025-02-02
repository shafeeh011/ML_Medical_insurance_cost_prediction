# Use Python 3.10 slim as base image
FROM python:3.10.6-slim

# Set working directory in the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Set the entry point to run the Streamlit app
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]


# terminal commands

# build the image
#sudo docker build -t ml_diabetes .

# run the image 
#sudo docker run -p 8000:8000 doker id

# get image id 
#sudo docker images

# running current container 
#sudo docker ps

# delete the container
#sudo docker rmi -f container_id