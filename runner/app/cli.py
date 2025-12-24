import argparse
import sys
from pathlib import Path

from app.models import Experiment
from app.experiment_runner import ExperimentRunner
from app.treatment_runner import LearnRunner
from app.io import upload_dir, download_by_key
from app.conf import BASE_RESULTS_DIR

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="causal-runner")
    p.add_argument("--experiment", required=True, help="Experiment JSON")
    p.add_argument(
        "--s3",
        required=True,
        help="S3 prefix s3://bucket/prefix",
    )
    p.add_argument("--skip-learn", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        exp_path = Path(args.experiment)
        exp = Experiment.model_validate_json(exp_path.read_text(encoding="utf-8"))
        base_dir = Path(BASE_RESULTS_DIR)

        base_dir.mkdir(parents=True, exist_ok=True)
        inputs_dir = base_dir / "inputs" / exp.name
        local_ds = inputs_dir / "dataset.csv"

        print(f"[runner-cli] downloading dataset key='{exp.ds_path}' from {args.s3} -> {local_ds}")
        download_by_key(
            s3_prefix=args.s3,
            key_no_prefix=exp.ds_path,
            dst_path=local_ds,
        )

        exp = exp.model_copy(update={"ds_path": str(local_ds)})

        discovery_runner = ExperimentRunner(base_results_dir=base_dir)
        learn_runner = LearnRunner(base_results_dir=base_dir)

        print(f"[runner-cli] experiment={exp.name}")
        print(f"[runner-cli] local_dataset={exp.ds_path}")

        discovery_summary = discovery_runner.run(exp)

        if not args.skip_learn and exp.causal_search:
            _ = learn_runner.run(exp)

        results_dir = Path(discovery_summary["results_dir"])
        uploaded, skipped = upload_dir(
            local_dir=results_dir,
            s3_url=args.s3,
            s3_subdir=f"{args.s3_subdir.strip('/')}/{exp.name}".strip("/"),
        )
        print(f"[runner-cli] upload: uploaded_files={uploaded}")

        print("[runner-cli] DONE")
        return 0

    except Exception as exc:
        print(f"[runner-cli] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
