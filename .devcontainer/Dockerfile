# Devcontainer configuration file for VSCode with Python 3.9 base image
FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip

# Install PyTorch (compatible version for CPU)
RUN pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 --index-url https://download.pytorch.org/whl/cpu

# Install Transformers and other dependencies
RUN pip install transformers==4.10.0 datasets==1.11.0 numpy==1.21.0 accelerate>=0.5.1 scikit-learn==0.24.2 pandas==1.3.3 seqeval

# Set the working directory
WORKDIR /workspace
