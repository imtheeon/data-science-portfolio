FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY model/ model/

EXPOSE 8000
USER nobody

# --forwarded-allow-ips="*" is required behind Render's load balancer: it is
# the only ingress to this container, but uvicorn's ProxyHeadersMiddleware by
# default only trusts X-Forwarded-For from 127.0.0.1. Without this flag every
# request appears to originate from the proxy's address, so slowapi's
# per-IP rate limit would apply globally across all visitors instead of per
# visitor. $PORT is the port Render assigns at runtime; it falls back to
# 8000 for a local `docker run` with no -e PORT set.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --forwarded-allow-ips=\"*\""]
