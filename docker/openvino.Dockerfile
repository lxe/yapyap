# Yap Yap server on Intel GPUs (Arc, Iris Xe, Core Ultra) through OpenVINO.
# The base image ships the Intel GPU compute runtime and OpenVINO GenAI.
FROM openvino/ubuntu24_dev:2026.4.1

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE /src/
COPY src /src/src
# OpenVINO GenAI comes with the base image; add the server and model download.
RUN python3 -m pip install --no-cache-dir --break-system-packages "/src[server]" "huggingface-hub>=2.4.0" \
    && rm -rf /src \
    && mkdir -p /models && chown openvino /models

USER openvino
ENV YAPYAP_ENGINE=openvino \
    YAPYAP_DEVICE=GPU \
    YAPYAP_HOST=0.0.0.0 \
    YAPYAP_PORT=8122 \
    YAPYAP_CACHE_DIR=/models
VOLUME /models
EXPOSE 8122
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8122/health', timeout=3)"
ENTRYPOINT ["yapyap-server"]
