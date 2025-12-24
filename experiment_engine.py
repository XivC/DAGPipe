import os
from dataclasses import dataclass
from typing import Any, Optional

import boto3
from celery import Celery

@dataclass
class S3Prefix:
    bucket: str
    prefix: str


def _parse_s3_prefix(s3_url: str) -> S3Prefix:
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Invalid s3_url")

    rest = s3_url[len("s3://") :].strip("/")
    if not rest:
        raise ValueError(f"Invalid s3_url: {s3_url}")

    parts = rest.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    prefix = prefix.strip("/")
    return S3Prefix(bucket=bucket, prefix=prefix)


def _join_key(prefix: str, key_no_prefix: str) -> str:
    key_no_prefix = key_no_prefix.lstrip("/")
    return f"{prefix}/{key_no_prefix}" if prefix else key_no_prefix


class ExperimentEngine:

    def __init__(self, queue_name: str = "engine.task", region_name: Optional[str] = None) -> None:
        self.queue_name = queue_name
        self.region_name = region_name
        self._celery: Optional[Celery] = None
        self._s3_url: Optional[str] = None
        self._s3_prefix: Optional[S3Prefix] = None
        self._s3_client = None

    def connect(self, celery_app: Celery, s3_url: str) -> None:
        self._celery = celery_app
        self._s3_url = s3_url
        self._s3_prefix = _parse_s3_prefix(s3_url)

        session = boto3.session.Session(region_name=self.region_name)
        self._s3_client = session.client("s3")

    def load_data(self, local_path: str, key_no_prefix: str) -> str:
        if not self._s3_client or not self._s3_prefix:
            raise RuntimeError("Engine is not connected. Call connect(celery_app, s3_url) first")

        if not os.path.exists(local_path):
            raise FileNotFoundError(local_path)
        key_no_prefix = key_no_prefix.lstrip("/")
        full_key = _join_key(self._s3_prefix.prefix, key_no_prefix)

        self._s3_client.upload_file(
            Filename=local_path,
            Bucket=self._s3_prefix.bucket,
            Key=full_key,
        )

        return key_no_prefix

    def run(self, experiment: dict[str, Any]) -> Any:

        if not self._celery or not self._s3_url:
            raise RuntimeError("Engine is not connected. Call connect(celery_app, s3_url) first")

        self._validate_experiment(experiment)

        payload = dict(experiment)
        payload["s3_prefix"] = self._s3_url

        async_result = self._celery.send_task(
            "engine.task",
            args=[payload],
            queue=self.queue_name,
        )
        return async_result

    @staticmethod
    def _validate_experiment(experiment: dict[str, Any]) -> None:
        required = ["name", "results_path", "ds_path", "methods"]
        for k in required:
            if k not in experiment:
                raise ValueError(f"Experiment missing required field: {k}")

        if not isinstance(experiment["methods"], list) or len(experiment["methods"]) == 0:
            raise ValueError("Experiment.methods must be a non-empty list")
