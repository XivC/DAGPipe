from celery import Celery
from app.conf import CONF

celery_app = Celery("orchestrator", broker=CONF.broker_url, backend=CONF.result_backend)

celery_app.conf.update(
    task_default_queue=CONF.queue_name,
)
