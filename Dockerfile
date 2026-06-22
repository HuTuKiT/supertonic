FROM python:3.11-slim

# Tránh Python tạo file .pyc và buffer output (tốt cho Docker logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Cài thư viện
RUN pip install --no-cache-dir 'supertonic[serve]'

# Download model vào image luôn (không cần download lại mỗi lần chạy)
RUN supertonic download

EXPOSE 7788

# Healthcheck: kiểm tra server có sống không mỗi 30s
# --start-period=60s: chờ 60s trước khi bắt đầu check (vì khởi động lâu)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7788/v1/health')" || exit 1

CMD ["supertonic", "serve", "--host", "0.0.0.0", "--port", "7788"]