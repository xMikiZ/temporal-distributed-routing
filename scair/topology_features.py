"""
Topology-derived feature vectors for GNN initialization and fixed no-GNN encodings.

All functions return a dict {node_id: torch.Tensor} of length feature_length,
zero-padded if the raw feature is shorter than feature_length.
"""

import math
from typing import Dict

import networkx as nx
import torch


def _build_nx(topo) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(topo.num_nodes))
    for u, neighbors in topo.adjacency.items():
        for v in neighbors:
            G.add_edge(u, v)
    return G


def _pad(raw: torch.Tensor, feature_length: int) -> torch.Tensor:
    v = torch.zeros(feature_length)
    n = min(len(raw), feature_length)
    v[:n] = raw[:n].float()
    return v


def compute_degree_init(topo, feature_length: int) -> Dict[int, torch.Tensor]:
    """V_n = [normalized_degree, 0, ..., 0]."""
    G = _build_nx(topo)
    max_deg = max(dict(G.degree()).values()) or 1
    result = {}
    for n in range(topo.num_nodes):
        raw = torch.tensor([G.degree(n) / max_deg])
        result[n] = _pad(raw, feature_length)
    return result


def compute_betweenness_init(topo, feature_length: int) -> Dict[int, torch.Tensor]:
    """V_n = [normalized_degree, betweenness_centrality, 0, ..., 0]."""
    G = _build_nx(topo)
    max_deg = max(dict(G.degree()).values()) or 1
    bc = nx.betweenness_centrality(G, normalized=True)
    result = {}
    for n in range(topo.num_nodes):
        raw = torch.tensor([G.degree(n) / max_deg, bc[n]])
        result[n] = _pad(raw, feature_length)
    return result


def compute_shortestpath_init(topo, feature_length: int) -> Dict[int, torch.Tensor]:
    """V_n = [dist_0, dist_1, ..., dist_{N-1}, 0, ..., 0], normalized by diameter."""
    G = _build_nx(topo)
    N = topo.num_nodes
    try:
        diameter = nx.diameter(G)
    except nx.NetworkXError:
        diameter = N
    diameter = max(diameter, 1)

    result = {}
    for n in range(N):
        lengths = nx.single_source_shortest_path_length(G, n)
        raw = torch.zeros(N)
        for dst, d in lengths.items():
            raw[dst] = d / diameter
        # unreachable nodes keep 0 (or set to 1 if you want max distance)
        result[n] = _pad(raw, feature_length)
    return result


def compute_init_vectors(topo, init_type: str,
                         feature_length: int) -> Dict[int, torch.Tensor]:
    """Dispatch to the right init function by name."""
    if init_type == "degree":
        return compute_degree_init(topo, feature_length)
    if init_type == "betweenness":
        return compute_betweenness_init(topo, feature_length)
    if init_type == "shortestpath":
        return compute_shortestpath_init(topo, feature_length)
    raise ValueError(f"Unknown init_type: {init_type!r}")
