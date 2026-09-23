FROM python:3.11-slim

WORKDIR /app

# 标准库实现，无需安装第三方依赖
COPY server.py schema.sql ./

# 持久化目录
VOLUME ["/data"]

ENV INFERENCE_DB=/data/inference.db \
    INFERENCE_STORAGE=/data/storage \
    INFERENCE_HOST=0.0.0.0 \
    INFERENCE_PORT=8090

EXPOSE 8090

CMD ["python", "server.py"]
