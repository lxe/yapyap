# Yap Yap server on CPUs, or NVIDIA GPUs with --build-arg CUDA=1, through whisper.cpp.
ARG CUDA=0
ARG BASE=python:3.12-slim-bookworm

FROM ${BASE}
ARG CUDA
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg build-essential cmake git \
       $( [ "$CUDA" = "1" ] && echo python3 python3-pip python3-venv ) \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE /src/
COPY src /src/src
# With CUDA=1, pywhispercpp is built from source against the CUDA toolkit.
RUN python3 -m venv /opt/yapyap \
    && if [ "$CUDA" = "1" ]; then \
         GGML_CUDA=1 /opt/yapyap/bin/pip install --no-cache-dir --no-binary pywhispercpp "/src[whispercpp]"; \
       else \
         /opt/yapyap/bin/pip install --no-cache-dir "/src[whispercpp]"; \
       fi \
    && rm -rf /src

ENV PATH=/opt/yapyap/bin:$PATH \
    YAPYAP_ENGINE=whispercpp \
    YAPYAP_HOST=0.0.0.0 \
    YAPYAP_PORT=8122 \
    XDG_DATA_HOME=/models
VOLUME /models
EXPOSE 8122
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8122/health', timeout=3)"
ENTRYPOINT ["yapyap-server"]
