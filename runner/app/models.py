from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Literal, Any
from pydantic import BaseModel, Field

class DAGMethod(str, Enum):
    PC = "PC"
    FCI = "FCI"
    GIN = "GIN"
    RESIT = "RESIT"
    BIC_EXACT_SEARCH = "BIC_EXACT_SEARCH"

    def __str__(self):
        return self.value


class ParamType(str, Enum):
    CATEGORICAL = "CATEGORICAL"
    NUMERICAL = "NUMERICAL"

class Param(NamedTuple):
    name: str
    min_: float | None = None
    max_: float | None = None
    step: float | None = None
    type_: ParamType | None = None


    def __str__(self):
        return self.name

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        return str(self) == str(other)

    def is_input(self) -> bool:
        return all([self.min_ is not None, self.max_ is not None, self.step is not None])


Edge = tuple[Param, Param]
DAG = set[Edge]

P = Param


@dataclass
class GraphMetrics:
    tp: int
    fp: int
    fn: int
    reversed: int
    precision: float
    recall: float
    f1: float
    shd: int  # structural Hamming distance



class VariableSpec(BaseModel):
    name: str
    type: ParamType


class CausalSearchSpec(BaseModel):
    treatment: list[VariableSpec]
    outcome: VariableSpec


class MethodSpec(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    timeout: int = 300  # seconds


class Experiment(BaseModel):
    name: str
    results_path: str
    ds_path: str
    methods: list[MethodSpec]
    causal_search: list[CausalSearchSpec] = Field(default_factory=list)
    true_dag_path: str | None = None
    columns: list[str] = None



@dataclass
class EffectEstimate:
    coef: float
    ci_low: float
    ci_high: float
    r2: float
    model: str