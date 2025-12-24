import os
from dataclasses import dataclass

@dataclass
class Conf:
    broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    queue_name: str = os.getenv("CELERY_QUEUE", "engine.task")
    backend: str = os.getenv("ORCH_BACKEND", "docker")
    runner_image: str = os.getenv("RUNNER_IMAGE", "causal-runner:dev")
    docker_host_url: str = os.getenv("DOCKER_HOST_URL", "unix:///var/run/docker.sock")
    k8s_namespace: str = os.getenv("K8S_NAMESPACE", "default")
    k8s_job_ttl_seconds: int = int(os.getenv("K8S_JOB_TTL_SECONDS", "3600"))
    k8s_pvc_claim: str = os.getenv("K8S_PVC_CLAIM", "")
    k8s_runner_out_mount: str = os.getenv("K8S_OUT_MOUNT", "/out")


CONF = Conf()
