import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from models import Experiment, MethodSpec
from models import P, Param, DAGMethod
from calculations import (
    GraphMetrics,
    evaluate_graph,
    edges_from_causallearn,
    edges_from_resit_matrix,
    edges_from_np,
)
from conf import BASE_RESULTS_DIR

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from lingam import RESIT
from sklearn.ensemble import RandomForestRegressor



def _load_params_from_dataset(df: pd.DataFrame, columns: Optional[list[str]]) -> list[Param]:
    cols = columns or list(df.columns)
    return [P(c) for c in cols]


def _load_true_dag(true_dag_path: str, params: list[Param]) -> set[tuple[Param, Param]]:
    pmap = {p.name: p for p in params}
    dag_df = pd.read_csv(true_dag_path)
    dag: set[tuple[Param, Param]] = set()
    for _, row in dag_df.iterrows():
        s = str(row["source"])
        t = str(row["target"])
        if s in pmap and t in pmap:
            dag.add((pmap[s], pmap[t]))
    return dag


def _default_resit_regressor_from_params(params: dict[str, Any]) -> RandomForestRegressor:
    max_depth = params.get("regressor__max_depth", 4)
    n_estimators = params.get("regressor__n_estimators", 100)
    return RandomForestRegressor(
        max_depth=int(max_depth),
        n_estimators=int(n_estimators),
        random_state=0,
        n_jobs=-1,
    )


class ExperimentRunner:
    def __init__(self, base_results_dir: Optional[Path] = None):
        self.base_results_dir = base_results_dir or Path(BASE_RESULTS_DIR)

    def run(self, experiment: Experiment) -> dict[str, Any]:
        df = pd.read_csv(experiment.ds_path)

        params = _load_params_from_dataset(df, experiment.columns)
        data = df[[p.name for p in params]].to_numpy()

        true_dag: Optional[set[tuple[Param, Param]]] = None
        if experiment.true_dag_path:
            true_dag = _load_true_dag(experiment.true_dag_path, params)

        exp_dir = self.base_results_dir / experiment.results_path / experiment.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "experiment.json").write_text(experiment.model_dump_json(indent=2), encoding="utf-8")

        job_summaries: list[dict[str, Any]] = []

        for m in experiment.methods:
            method_enum = DAGMethod(m.name)
            method_dir = exp_dir / m.name
            method_dir.mkdir(parents=True, exist_ok=True)

            predicted, metrics = self._run_one_method(
                method=method_enum,
                method_spec=m,
                data=data,
                params=params,
                true_dag=true_dag,
            )

            self._save_method_results(
                method_dir=method_dir,
                predicted_dag=predicted,
                true_dag=true_dag,
                metrics=metrics,
            )

            job_summaries.append({
                "method": m.name,
                "has_true_dag": bool(true_dag),
                "metrics": (metrics.__dict__ if metrics else None),
            })

        (exp_dir / "causal_search.json").write_text(
            json.dumps([q.model_dump() for q in experiment.causal_search], indent=2),
            encoding="utf-8"
        )

        summary = {
            "experiment": experiment.name,
            "results_dir": str(exp_dir),
            "methods": job_summaries,
        }
        (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return summary

    def _run_one_method(
        self,
        *,
        method: DAGMethod,
        method_spec: MethodSpec,
        data,
        params: list[Param],
        true_dag: Optional[set[tuple[Param, Param]]],
    ) -> tuple[set[tuple[Param, Param]], Optional[GraphMetrics]]:
        if method == DAGMethod.PC:
            alpha = float(method_spec.params.get("alpha", 0.1))
            cg = pc(data, alpha=alpha)
            predicted = edges_from_causallearn(cg.G, params)

        elif method == DAGMethod.FCI:
            alpha = float(method_spec.params.get("alpha", 0.1))
            g, _ = fci(data, alpha=alpha, independence_test_method="fisherz", depth=-1, max_path_length=-1)
            predicted = edges_from_causallearn(g, params)

        elif method == DAGMethod.GIN:
            alpha = float(method_spec.params.get("alpha", 0.1))
            g, _ = GIN(data, indep_test_method="kci", alpha=alpha)
            predicted = edges_from_causallearn(g, params)

        elif method == DAGMethod.BIC_EXACT_SEARCH:
            bes_params = {
                "super_graph": None,
                "search_method": "astar",
                "use_path_extension": True,
                "use_k_cycle_heuristic": False,
                "k": 3,
                "verbose": True,
                "max_parents": None,
            }
            bes_params.update(method_spec.params or {})
            g, _ = bic_exact_search(data, **bes_params)
            predicted = edges_from_np(g, params)

        elif method == DAGMethod.RESIT:
            regressor_name = method_spec.params.get("regressor", "RandomForestRegressor")

            if regressor_name != "RandomForestRegressor":
                raise ValueError(f"Only RandomForestRegressor supported in prototype, got: {regressor_name}")

            reg = _default_resit_regressor_from_params(method_spec.params)
            model = RESIT(regressor=reg)
            model.fit(data)
            W = model.adjacency_matrix_
            predicted = edges_from_resit_matrix(W, params)

        else:
            raise ValueError(f"Unsupported method: {method}")

        metrics = evaluate_graph(true_dag, predicted) if true_dag is not None else None
        return predicted, metrics

    def _save_method_results(
        self,
        *,
        method_dir: Path,
        predicted_dag: set[tuple[Param, Param]],
        true_dag: Optional[set[tuple[Param, Param]]],
        metrics: Optional[GraphMetrics],
    ) -> None:
        def dag_to_df(dag: set[tuple[Param, Param]]) -> pd.DataFrame:
            return pd.DataFrame([{"source": s.name, "target": t.name} for s, t in dag])

        dag_to_df(predicted_dag).to_csv(method_dir / "predicted_dag.csv", index=False)

        if true_dag is not None:
            dag_to_df(true_dag).to_csv(method_dir / "true_dag.csv", index=False)

        if metrics is not None:
            pd.DataFrame([metrics]).to_csv(method_dir / "metrics.csv", index=False)
        else:
            (method_dir / "metrics.csv").write_text("", encoding="utf-8")
