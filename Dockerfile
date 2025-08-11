# Use an official NVIDIA CUDA image as the base.
# This image includes the CUDA Toolkit, NVCC, and necessary drivers.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set the frontend to non-interactive to avoid prompts during package installation.
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust using rustup in a non-interactive way.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add cargo to the PATH environment variable.
ENV PATH="/root/.cargo/bin:${PATH}"

# Create a directory for the application source code.
WORKDIR /app

# Copy the application source code into the container.
COPY . .

# Set the default command to execute the build script.
# This will be run when the container starts.
CMD ["/bin/bash", "build.sh"]
