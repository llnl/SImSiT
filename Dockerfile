FROM python:3.10

RUN apt-get update -y && \
    apt-get install -y \
        git \
        git-lfs \
        wget \
        build-essential \
        gcc \
        cmake && \
    rm -rf /var/lib/apt/lists/*

RUN wget -c http://www.fftw.org/fftw-3.3.10.tar.gz -O - | tar -xz && \
    cd fftw-3.3.10 && \
    ./configure && \
    make && \
    make install && \
    make check

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python -m pip install --upgrade pip

COPY . /src
WORKDIR /src

RUN python -m pip install '.[dev,docs]'