import json
import os
import time
from dataclasses import dataclass
from typing import Any

from .conf import CONF
from kubernetes import client, config
import docker

@dataclass
class JobResult:
    job_id: str
    method: str
    exit_code: int


class BackendBase:
    def run_many(self, jobs: list[dict[str, Any]]) -> list[JobResult]:
        raise NotImplementedError


class DockerBackend(BackendBase):
    def __init__(self, runner_image: str, docker_host_url: str):
        self.docker = docker.DockerClient(base_url=docker_host_url)
        self.runner_image = runner_image

    def _runner_env(self) -> dict[str, str]:
        env = {"BASE_RESULTS_DIR": "/out"}
        for k, v in os.environ.items():
            if k.startswith("AWS_"):
                env[k] = v
        return env

    def _run_one(self, exp_json: dict[str, Any]) -> JobResult:
        exp_name = exp_json.get("name", "experiment")
        method = exp_json["methods"][0]["name"]
        job_id = f"{exp_name}"
        exp_str = json.dumps(exp_json, ensure_ascii=False)
        s3_prefix = exp_json.get("s3_prefix") or exp_json.get("s3")

        cmd = [
            "--experiment", exp_str,
            "--s3", str(s3_prefix),
        ]

        container = self.docker.containers.run(
            self.runner_image,
            command=cmd,
            environment=self._runner_env(),
            detach=True,
        )

        try:
            timeout_s = int(exp_json["methods"][0].get("timeout", 3600))
            res = container.wait(timeout=timeout_s)
            exit_code = int(res.get("StatusCode", 1))
        except Exception:
            try:
                container.stop(timeout=3)
            except Exception:
                pass
            exit_code = 1
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass
            exit_code = 0

        return JobResult(job_id=job_id, method=method, exit_code=exit_code)

    def run_many(self, jobs: list[dict[str, Any]]) -> list[JobResult]:
        results: list[JobResult] = []
        for exp_json in jobs:
            results.append(self._run_one(exp_json))
        return results

class K8sBackend(BackendBase):

    def __init__(self, runner_image: str, namespace: str, base_results_dir_mount: str):

        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()

        self.k8s_client = client
        self.api = client.BatchV1Api()
        self.core = client.CoreV1Api()
        self.runner_image = runner_image
        self.namespace = namespace
        self.out_mount = base_results_dir_mount

    def _aws_env_vars(self):
        env = [self.k8s_client.V1EnvVar(name="BASE_RESULTS_DIR", value=self.out_mount)]
        for k, v in os.environ.items():
            if k.startswith("AWS_"):
                env.append(self.k8s_client.V1EnvVar(name=k, value=v))
        return env

    def _build_job(self, name: str, exp_json: dict[str, Any]) -> Any:
        method = exp_json["methods"][0]["name"]
        exp_str = json.dumps(exp_json, ensure_ascii=False)

        s3_prefix = exp_json.get("s3_prefix") or exp_json.get("s3")

        args = [
            "--experiment", exp_str,
            "--s3", str(s3_prefix),
        ]

        container = self.k8s_client.V1Container(
            name="runner",
            image=self.runner_image,
            args=args,
            env=self._aws_env_vars(),
        )

        pod_spec = self.k8s_client.V1PodSpec(
            restart_policy="Never",
            containers=[container],
        )

        tpl = self.k8s_client.V1PodTemplateSpec(
            metadata=self.k8s_client.V1ObjectMeta(labels={"app": "causal-runner", "method": method}),
            spec=pod_spec,
        )

        job_spec = self.k8s_client.V1JobSpec(
            template=tpl,
            backoff_limit=0,
            ttl_seconds_after_finished=CONF.k8s_job_ttl_seconds,
        )

        job = self.k8s_client.V1Job(
            metadata=self.k8s_client.V1ObjectMeta(name=name),
            spec=job_spec,
        )
        return job

    def _wait_job(self, job_name: str, timeout_s: int):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            j = self.api.read_namespaced_job_status(job_name, self.namespace)
            st = j.status
            if st.succeeded and st.succeeded >= 1:
                return 0
            if st.failed and st.failed >= 1:
                return 1
            time.sleep(1)


    def run_many(self, jobs: list[dict[str, Any]]) -> list[JobResult]:
        results: list[JobResult] = []
        for exp_json in jobs:
            exp_name = exp_json.get("name", "experiment").lower().replace("_", "-")
            method = exp_json["methods"][0]["name"].lower().replace("_", "-")
            job_name = f"{exp_name}-{method}-{int(time.time())}"

            job_obj = self._build_job(job_name, exp_json)
            self.api.create_namespaced_job(namespace=self.namespace, body=job_obj)

            timeout_s = int(exp_json["methods"][0].get("timeout", 3600))
            exit_code = self._wait_job(job_name, timeout_s)

            results.append(JobResult(job_id=job_name, method=exp_json["methods"][0]["name"], exit_code=exit_code))
        return results


def get_backend() -> BackendBase:
    if CONF.backend == "docker":
        return DockerBackend(
            runner_image=CONF.runner_image,
            docker_host_url=CONF.docker_host_url,
        )
    return K8sBackend(
        runner_image=CONF.runner_image,
        namespace=CONF.k8s_namespace,
        base_results_dir_mount=CONF.k8s_runner_out_mount,
    )
