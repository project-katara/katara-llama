# syntax=docker/dockerfile:1
# Build as `docker build . -t localgpt`, requires BuildKit.
# Run as `docker run -it --mount src="$HOME/.cache",target=/root/.cache,type=bind --gpus=all localgpt`, requires Nvidia container toolkit.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential libpq-dev gcc \
    wget ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev software-properties-common \
    g++-11 make python3 python-is-python3 pip redis \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

COPY . .

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install depencencies
RUN python -m pip install --upgrade pip pytest cmake \
    scikit-build setuptools fastapi uvicorn sse-starlette \
    pydantic-settings starlette-context gradio huggingface_hub hf_transfer

RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 PIP_ROOT_USER_ACTION=ignore pip install --ignore-installed --timeout 100 -r requirements.txt
RUN pip install uvicorn

RUN useradd -m -u 1000 user
USER user

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

RUN chmod +x ./run.sh

CMD ["sh", "run.sh"]
