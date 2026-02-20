FROM python:3.11-slim

WORKDIR /app

# System dependencies for nginx + supervisor + envsubst
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx supervisor curl gettext-base \
  && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY . /app

# Nginx config template (PORT substituted at runtime)
COPY nginx.conf /etc/nginx/nginx.conf

# Supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Render expects one process bound to $PORT
CMD ["/bin/bash", "-lc", "envsubst '$PORT' < /etc/nginx/nginx.conf > /tmp/nginx.conf && nginx -c /tmp/nginx.conf && supervisord -c /etc/supervisor/conf.d/supervisord.conf"]
