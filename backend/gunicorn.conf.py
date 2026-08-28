import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
workers = 1
threads = 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
max_requests = 100
max_requests_jitter = 20
graceful_timeout = 30
accesslog = "-"
errorlog = "-"
preload_app = False
