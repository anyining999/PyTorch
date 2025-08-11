# RedcodE: A High-Performance Deep Learning Framework

This repository contains the source code for RedcodE, a revolutionary deep learning framework designed with a three-layer hybrid architecture. It combines the performance of C++ and CUDA with the safety and modern concurrency of Rust.

## Core Design Philosophy

- **Mathematical Simplicity:** Based on the golden ratio φ=0.618.
- **Extreme Performance:** C++17 core with CUDA acceleration and Rust scheduling.
- **Engineering Practicality:** Designed to solve real-world engineering problems.
- **Ecosystem-Friendly:** Aims for seamless integration with PyTorch and TensorFlow.

## Building the Project

This project uses a containerized build environment to ensure all dependencies are met and the build is reproducible.

### Prerequisites

- [Docker](https://www.docker.com/get-started)

### Build and Test Instructions

1.  **Build the Docker Image:**
    From the root of the repository, run the following command. This will build the Docker image, which contains all necessary dependencies (CUDA, Rust, CMake) and compiles the project.

    ```sh
    docker build -t redcode-build .
    ```

2.  **Run the Build and Tests:**
    The build and tests are run automatically when building the image. The `CMD` instruction in the `Dockerfile` executes the `build.sh` script. You can view the output of the build and tests during the `docker build` process.

    If you wish to run the build process again inside a container, you can do so with:

    ```sh
    docker run --rm redcode-build
    ```