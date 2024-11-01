FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip

# Install PyTorch (compatible version for CPU)
RUN pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 --index-url https://download.pytorch.org/whl/cpu

# Install Transformers and other dependencies
RUN pip install transformers==4.10.0 datasets==1.11.0 numpy==1.21.0 accelerate>=0.5.1

# Set the working directory
WORKDIR /workspace
