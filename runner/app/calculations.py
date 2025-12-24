import numpy as np
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.NodeType import NodeType

from app.models import Param, DAG, GraphMetrics


def edges_from_causallearn(cl_graph, params: list[Param]) -> DAG:

    nodes = cl_graph.get_nodes()
    origin_nodes = [node for node in nodes if node.node_type == NodeType.MEASURED]

    node_to_param = {node: params[i] for i, node in enumerate(origin_nodes)}

    def _ntp(node: GraphNode) -> Param:
        return node_to_param.get(node, Param(node.name))
    result = set()

    for edge in cl_graph.get_graph_edges():
        n1 = edge.get_node1()
        n2 = edge.get_node2()
        e1 = edge.get_endpoint1()
        e2 = edge.get_endpoint2()

        if e1 == Endpoint.TAIL and e2 == Endpoint.ARROW:
            result.add((_ntp(n1), _ntp(n2)))

        elif e1 == Endpoint.ARROW and e2 == Endpoint.TAIL:
            result.add((_ntp(n2), _ntp(n1)))

    return result


def edges_from_np(cl_graph: np.ndarray, params: list[Param]) -> DAG:

    result: DAG = set()

    n = len(cl_graph)
    for u in range(n):
        row = cl_graph[u]
        for v in range(n):
            if row[v]:
                result.add((params[u], params[v]))
    return result



def edges_from_resit_matrix(W: np.ndarray, params: list[Param]) -> set[tuple[Param, Param]]:
    d = W.shape[0]
    result: set[tuple[Param, Param]] = set()

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            wij = W[i, j]
            if wij is not None and not np.isnan(wij) and wij != 0:
                result.add((params[i], params[j]))

    return result


def evaluate_graph(expected: set[tuple[Param, Param]],
                   predicted: set[tuple[Param, Param]]) -> GraphMetrics:
    tp_edges = expected & predicted
    fp_edges = predicted - expected
    fn_edges = expected - predicted

    reversed_edges = set()
    for (u, v) in expected:
        if (v, u) in predicted:
            reversed_edges.add((u, v))

    tp = len(tp_edges)
    fp = len(fp_edges)
    fn = len(fn_edges)
    rev = len(reversed_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    shd = fp + fn - rev

    return GraphMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        reversed=rev,
        precision=precision,
        recall=recall,
        f1=f1,
        shd=shd,
    )
