import copy
import logging
from typing import Any

from app.celery_app import celery_app
from app.backends import get_backend
from app.conf import CONF

log = logging.getLogger("orchestrator")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


def split_into_method_runs(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    methods = experiment.get("methods") or []
    if not methods:
        raise ValueError("Experiment has no methods")

    base_name = experiment.get("name", "Experiment")
    out: list[dict[str, Any]] = []

    for m in methods:
        sub = copy.deepcopy(experiment)
        sub["methods"] = [m]

        method_name = m.get("name", "method")
        sub["name"] = f"{base_name}__{method_name}"
        out.append(sub)

    return out


@celery_app.task(name="engine.task", queue=CONF.queue_name)
def engine_task(experiment: dict):

    exp_name = experiment.get("name", "Experiment")
    dataset = experiment.get("ds_path")

    log.info("Received experiment task from ExperimentEngine")
    log.info("Experiment name=%s dataset=%s", exp_name, dataset)

    methods = [m["name"] for m in experiment.get("methods", [])]
    log.info("Methods to run: %s", methods)

    if experiment.get("causal_search"):
        q = experiment["causal_search"][0]
        t = q["treatment"][0]["name"]
        y = q["outcome"]["name"]
        log.info("Causal query: treatment=%s outcome=%s", t, y)

    log.info("")
    log.info("Creating jobs for experiment %s", exp_name)

    sub_experiments = split_into_method_runs(experiment)

    jobs = []
    for i, sub in enumerate(sub_experiments, start=1):
        method = sub["methods"][0]["name"]
        timeout = sub["methods"][0].get("timeout", 300)
        job_id = f"job-{i}"
        jobs.append((job_id, sub))

        log.info(
            "Job created: job_id=%s method=%s timeout=%ss",
            job_id,
            method,
            timeout,
        )

    log.info("")
    backend = get_backend()

    results = []
    for job_id, sub_exp in jobs:
        log.info("Dispatching job %s to runner", job_id)
        log.info("Job %s started (backend=%s)", job_id, CONF.backend)

        res = backend.run_many([sub_exp])[0]
        results.append(res)

        status = "SUCCESS" if res.exit_code == 0 else "FAILED"
        log.info("Job %s finished with status=%s", job_id, status)

    log.info("All jobs for experiment %s completed", exp_name)

    final_status = "successfully" if all(r.exit_code == 0 for r in results) else "with errors"
    log.info("Experiment %s finished %s", exp_name, final_status)

    return {
        "experiment": exp_name,
        "status": "SUCCESS" if final_status == "successfully" else "FAILED",
    }
