FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg \
        lsb-release \
        pkg-config \
        rsync \
        software-properties-common \
        tar \
        unzip \
        wget \
        zip \
        build-essential \
        ninja-build \
        cmake \
        clang-tidy \
        libvulkan-dev \
        xorg-dev \
        libxinerama-dev \
        libxcursor-dev \
        libglu1-mesa-dev \
    && wget -qO - https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-22 main" \
        > /etc/apt/sources.list.d/llvm-toolchain-noble-22.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends clang-format-22 \
    && rm -rf /var/lib/apt/lists/*

COPY ci-stages.sh /usr/local/bin/ci-stages.sh
COPY container-run.sh /usr/local/bin/fireengine-ci

WORKDIR /work/fireEngine
