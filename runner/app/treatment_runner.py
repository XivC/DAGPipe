from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.lines import Line2D
from pandas.errors import EmptyDataError

from app.models import Param, P, ParamType, EffectEstimate
from app.models import Experiment, VariableSpec



def _pname(p: Param | str) -> str:
    return p if isinstance(p, str) else p.name


def _to_param(v: VariableSpec) -> Param:
    t = v.type
    if t == "CATEGORICAL":
        type_ = ParamType.CATEGORICAL
    else:
        type_ = ParamType.NUMERICAL
    return P(v.name, type_=type_)


def _read_dag_csv(path: Path) -> nx.DiGraph | None:
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, EmptyDataError):
        return None

    g = nx.DiGraph()
    for _, r in df.iterrows():
        g.add_edge(str(r["source"]), str(r["target"]))
    return g


class LearnEngine:
    def __init__(self, base_results_dir: Path):
        self.base_results_dir = base_results_dir


    def exp_dir(self, experiment: Experiment) -> Path:
        p = self.base_results_dir / experiment.results_path / experiment.name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def method_dir(self, experiment: Experiment, method_name: str) -> Path:
        p = self.exp_dir(experiment) / method_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def effects_dir(self, experiment: Experiment, method_name: str, effect_name: str) -> Path:
        p = self.exp_dir(experiment) / "effects" / method_name / effect_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def load_predicted_dag(self, experiment: Experiment, method_name: str) -> nx.DiGraph | None:
        path = self.method_dir(experiment, method_name) / "predicted_dag.csv"
        return _read_dag_csv(path)

    def load_true_dag(self, experiment: Experiment) -> nx.DiGraph | None:
        if not experiment.true_dag_path:
            return None
        return _read_dag_csv(Path(experiment.true_dag_path))

    def backdoor_adjustment_set(self, g: nx.DiGraph, treatment: list[Param], outcome: Param) -> list[str]:
        T = [_pname(t) for t in treatment]
        Y = _pname(outcome)

        cand = set()
        for t in T:
            cand |= set(g.predecessors(t))

        forbidden = set(T) | {Y}
        for t in T:
            forbidden |= nx.descendants(g, t)
        forbidden |= nx.descendants(g, Y)

        adj = sorted(c for c in cand if c not in forbidden)

        # optional minimality
        if hasattr(nx, "d_separated"):
            Z = set(adj)
            changed = True
            while changed:
                changed = False
                for z in list(Z):
                    Z_try = Z - {z}
                    if nx.d_separated(g, set(T), {Y}, Z_try):
                        Z = Z_try
                        changed = True
            adj = sorted(Z)

        return adj

    def build_feature_sets(
        self,
        df: pd.DataFrame,
        g: nx.DiGraph | None,
        treatment: list[Param],
        outcome: Param,
    ) -> dict[str, list[str]]:
        Y = _pname(outcome)
        T = [_pname(t) for t in treatment]

        naive = [c for c in df.columns if c != Y]

        if g is None:
            return {
                "naive": [c for c in naive if c in df.columns],
                "causal": [c for c in naive if c in df.columns],
            }

        adj = self.backdoor_adjustment_set(g, treatment, outcome)
        causal = list(dict.fromkeys(T + adj))

        return {
            "naive": [c for c in naive if c in df.columns],
            "causal": [c for c in causal if c in df.columns],
        }

    def _select_model(self, treatment: list[Param], outcome: Param) -> str:
        t = treatment[0]
        y = outcome
        t_type = getattr(t, "type_", None)
        y_type = getattr(y, "type_", None)

        if y_type == ParamType.NUMERICAL and t_type == ParamType.NUMERICAL:
            return "OLS"

        if y_type == ParamType.CATEGORICAL and t_type == ParamType.NUMERICAL:
            return "LOGIT"

        if y_type == ParamType.NUMERICAL and t_type == ParamType.CATEGORICAL:
            return "OLS_DUMMY"

        if y_type == ParamType.CATEGORICAL and t_type ==  ParamType.CATEGORICAL:
            return "LOGIT_DUMMY"

        raise ValueError(f"Unsupported (treatment, outcome) types: {getattr(t, 'type_', None)}, {getattr(y, 'type_', None)}")

    def _encode_X(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        cat_cols = []
        for c in features:
            if c not in df.columns:
                continue
            dt = df[c].dtype
            if dt == "object" or str(dt).startswith("category") or str(dt) == "bool":
                cat_cols.append(c)

        return pd.get_dummies(df[features], columns=cat_cols, drop_first=True)

    def estimate(self, df: pd.DataFrame, features: list[str], treatment: list[Param], outcome: Param) -> EffectEstimate:
        model = self._select_model(treatment, outcome)

        Y = _pname(outcome)
        T0 = _pname(treatment[0])

        if model in {"OLS", "LOGIT"}:
            X = df[features]
        else:
            X = self._encode_X(df, features)

        X = sm.add_constant(X, has_constant="add")
        y = df[Y]

        if model.startswith("OLS"):
            m = sm.OLS(y, X).fit(cov_type="HC1")
            coef = float(m.params[T0])
            ci_low, ci_high = m.conf_int().loc[T0]
            r2 = float(m.rsquared)

        elif model.startswith("LOGIT"):
            y = y.astype(int)
            m = sm.Logit(y, X).fit(disp=False)
            coef = float(m.params[T0])
            ci_low, ci_high = m.conf_int().loc[T0]
            r2 = float(1.0 - m.llf / m.llnull)  # McFadden pseudo-R2

        else:
            raise RuntimeError("Unknown model")

        return EffectEstimate(coef=float(coef), ci_low=float(ci_low), ci_high=float(ci_high), r2=float(r2), model=model)

    def plot_effect_comparison(self, rows: list[dict], out: Path, title: str):
        modes = [r["mode"] for r in rows]
        coefs = [r["coef"] for r in rows]
        ci_low = [r["ci_low"] for r in rows]
        ci_high = [r["ci_high"] for r in rows]

        y = np.arange(len(rows))
        plt.figure(figsize=(8, 3.5))

        for i in range(len(rows)):
            plt.plot([ci_low[i], ci_high[i]], [y[i], y[i]], color="black", lw=2)

        plt.scatter(coefs, y, color="blue", zorder=3)
        plt.axvline(0.0, color="gray", linestyle="--", lw=1)

        plt.yticks(y, modes)
        plt.xlabel("Treatment effect (coef / log-odds)")
        plt.title(title)
        plt.grid(axis="x", alpha=0.3)

        legend_items = [
            Line2D([0], [0], marker='o', color='blue', label='Point estimate', linestyle='None'),
            Line2D([0], [0], color='black', lw=2, label='95% CI'),
            Line2D([0], [0], color='gray', lw=1, linestyle='--', label='No effect'),
        ]
        plt.legend(handles=legend_items, loc="lower right", frameon=True)

        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()

    def plot_dag_roles(self, g: nx.DiGraph, adj: list[str], treatment: list[Param], outcome: Param, out: Path, title: str):
        T = {_pname(t) for t in treatment}
        Y = _pname(outcome)
        Z = set(adj)

        colors = []
        for n in g.nodes():
            if n in T:
                colors.append("orange")
            elif n == Y:
                colors.append("red")
            elif n in Z:
                colors.append("green")
            else:
                colors.append("lightgray")

        pos = nx.spring_layout(g, seed=42)
        plt.figure(figsize=(7, 6))
        nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=900)
        nx.draw_networkx_edges(g, pos, arrows=True)
        nx.draw_networkx_labels(g, pos)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()

    def run_effect(
        self,
        *,
        experiment: Experiment,
        method_name: str,
        treatment: list[Param],
        outcome: Param,
    ) -> dict:
        df = pd.read_csv(experiment.ds_path)

        g_true = self.load_true_dag(experiment)
        g_pred = self.load_predicted_dag(experiment, method_name)

        rows: list[dict] = []
        effect_name = f"{'_'.join(_pname(t) for t in treatment)}__{_pname(outcome)}"
        out_dir = self.effects_dir(experiment, method_name, effect_name)

        def row(mode: str, g: nx.DiGraph | None) -> dict:
            feats = self.build_feature_sets(df, g, treatment, outcome)
            est = self.estimate(df, feats["causal"], treatment, outcome)

            if mode == "NAIVE" or g is None or g.size(None) == 0:
                adj = []
            else:
                adj = self.backdoor_adjustment_set(g, treatment, outcome)

            return {
                "method": method_name,
                "mode": mode,  # NAIVE / PREDICTED_DAG / TRUE_DAG
                "treatment": ",".join(_pname(t) for t in treatment),
                "outcome": _pname(outcome),
                "adjustment_set": "|".join(adj),
                "coef": est.coef,
                "ci_low": est.ci_low,
                "ci_high": est.ci_high,
                "r2": est.r2,
                "model": est.model,
            }
        rows.append(row("NAIVE", None))

        if g_pred is not None and len(g_pred.edges()) > 0:
            rows.append(row("PREDICTED_DAG", g_pred))
        else:
            pass
        if g_true is not None and len(g_true.edges()) > 0:
            rows.append(row("TRUE_DAG", g_true))

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(out_dir / "summary.csv", index=False)

        # plots
        title = rows[0]["model"]
        self.plot_effect_comparison(rows, out_dir / "effect_comparison.png", title=title)

        if g_pred is not None and len(g_pred.nodes()) > 0:
            adj = self.backdoor_adjustment_set(g_pred, treatment, outcome) if len(g_pred.edges()) > 0 else []
            self.plot_dag_roles(g_pred, adj, treatment, outcome, out_dir / "dag_roles_predicted.png", title="Predicted DAG (roles)")

        if g_true is not None and len(g_true.nodes()) > 0:
            adj = self.backdoor_adjustment_set(g_true, treatment, outcome) if len(g_true.edges()) > 0 else []
            self.plot_dag_roles(g_true, adj, treatment, outcome, out_dir / "dag_roles_true.png", title="True DAG (roles)")

        return {
            "effect": effect_name,
            "out_dir": str(out_dir),
            "rows": rows,
        }


class LearnRunner:
    def __init__(self, base_results_dir: Path):
        self.engine = LearnEngine(base_results_dir=base_results_dir)

    def run(self, experiment: Experiment) -> dict[str, Any]:
        exp_dir = self.engine.exp_dir(experiment)

        summaries: list[dict] = []

        for m in experiment.methods:
            method_name = m.name
            for q in experiment.causal_search:
                treatment = [_to_param(v) for v in q.treatment]
                outcome = _to_param(q.outcome)

                res = self.engine.run_effect(
                    experiment=experiment,
                    method_name=method_name,
                    treatment=treatment,
                    outcome=outcome,
                )
                summaries.append({
                    "method": method_name,
                    "effect": res["effect"],
                    "out_dir": res["out_dir"],
                })

        summary_path = exp_dir / "effects" / "effects_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

        return {"experiment": experiment.name, "effects": summaries}