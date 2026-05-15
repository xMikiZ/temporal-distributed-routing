"""
Data loading for ScaIR.

Handles:
  - Topology files (Abilene format: "num_nodes num_links\\n u v ...")
  - Traffic-matrix files (.dat, CSV with '#' comment lines, units Gbytes/s)
  - link_weight.json (optional per-directed-link delay overrides)

The paper uses the Abilene (11 nodes, 14 links) and GEANT (23 nodes, 37 links)
topologies with their real-world traffic matrices from the Abilene dataset.
"""

import json
import os
import glob
import xml.etree.ElementTree as ET
from math import gcd
from functools import reduce
from typing import Dict, List, Optional, Tuple

import numpy as np

from .environment import Topology


# ---------------------------------------------------------------------------
# Topology loader
# ---------------------------------------------------------------------------

def load_topology(
    topology_file: str,
    link_weight_file: Optional[str] = None,
    transmission_time: float = 1.0,
) -> Topology:
    """
    Load topology from an Abilene-style text file.

    File format:
        Line 0:  num_nodes  num_links
        Lines 1..num_links:  u  v  [distance  bandwidth  delay  ...]  (1-indexed)
        Last line: per-node capacity or similar (ignored)

    All link delays default to `transmission_time` ms (paper §5.1: fixed 1 ms).
    If link_weight_file is given, those weights override the default.
    The weights in link_weight.json correspond to directed edges in the order
    they are encountered while iterating links in both directions.
    """
    with open(topology_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    first = lines[0].split()
    num_nodes = int(first[0])
    num_links = int(first[1])

    adjacency: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    link_delays: Dict[Tuple[int, int], float] = {}
    edges: List[Tuple[int, int]] = []  # ordered directed edges (both directions)

    for i in range(1, num_links + 1):
        if i >= len(lines):
            break
        parts = lines[i].split()
        u, v = int(parts[0]) - 1, int(parts[1]) - 1  # convert to 0-indexed

        adjacency[u].append(v)
        adjacency[v].append(u)

        link_delays[(u, v)] = transmission_time
        link_delays[(v, u)] = transmission_time

        edges.append((u, v))
        edges.append((v, u))

    # Override delays with per-link weights if provided
    if link_weight_file and os.path.exists(link_weight_file):
        with open(link_weight_file) as f:
            weights = json.load(f)  # list of floats, one per directed edge
        for idx, (u, v) in enumerate(edges):
            if idx < len(weights):
                link_delays[(u, v)] = float(weights[idx])

    return Topology(num_nodes, adjacency, link_delays)


# ---------------------------------------------------------------------------
# Traffic matrix loader
# ---------------------------------------------------------------------------

def load_traffic_matrix(tm_file: str, num_nodes: int) -> np.ndarray:
    """
    Load a single traffic matrix file.

    Supports:
      - CSV with '#' comment lines (Abilene .dat format, units: Gbytes/s)
        The file may contain more rows/cols than `num_nodes`; we take the
        top-left num_nodes × num_nodes sub-matrix.
      - Plain numpy .npy files
      - Plain text matrices (space or comma separated, no header)

    Returns: (num_nodes, num_nodes) float64 array with diagonal zeroed.
    """
    ext = os.path.splitext(tm_file)[1].lower()

    if ext == ".npy":
        tm = np.load(tm_file).astype(np.float64)
    elif ext == ".xml":
        # GEANT XML format: <src id="X"><dst id="Y">value</dst>...</src>
        # Node IDs are 1-indexed; we convert to 0-indexed.
        tree = ET.parse(tm_file)
        root = tree.getroot()
        # Find the IntraTM element (ignore namespace variations)
        values: Dict[Tuple[int, int], float] = {}
        max_id = 0
        for src_el in root.iter("src"):
            src_id = int(src_el.get("id")) - 1   # 0-indexed
            max_id = max(max_id, src_id)
            for dst_el in src_el:
                dst_id = int(dst_el.get("id")) - 1
                max_id = max(max_id, dst_id)
                values[(src_id, dst_id)] = float(dst_el.text)
        size = max_id + 1
        tm = np.zeros((size, size), dtype=np.float64)
        for (i, j), v in values.items():
            tm[i, j] = v
    else:
        # Try text/CSV with possible '#' comments
        rows: List[List[float]] = []
        with open(tm_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Accept both comma and whitespace delimiters
                parts = line.replace(",", " ").split()
                rows.append([float(x) for x in parts])

        if not rows:
            raise ValueError(f"No data found in {tm_file}")

        tm = np.array(rows, dtype=np.float64)

    # Take top-left submatrix for the actual topology size
    n = min(num_nodes, tm.shape[0], tm.shape[1])
    tm_out = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    tm_out[:n, :n] = tm[:n, :n]
    np.fill_diagonal(tm_out, 0.0)

    return tm_out


def load_all_traffic_matrices(tm_dir: str, num_nodes: int) -> List[np.ndarray]:
    """
    Load all .dat (and .npy / .txt) traffic matrix files from a directory.
    Returns a list of (num_nodes, num_nodes) arrays, sorted by filename.
    """
    patterns = ["*.dat", "*.npy", "*.txt", "*.xml"]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(tm_dir, pat)))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No traffic matrix files found in {tm_dir}")

    matrices = [load_traffic_matrix(f, num_nodes) for f in files]
    return matrices


# ---------------------------------------------------------------------------
# Convenience: normalise traffic matrix for packet generation
# ---------------------------------------------------------------------------

def normalise_tm(tm: np.ndarray) -> np.ndarray:
    """
    Convert traffic rates to exact integer weights preserving all pairwise
    proportions (paper §5.1: "integer processing in accordance with the same
    proportion").

    Dividing by the minimum non-zero value ensures every non-zero (i,j) pair
    maps to at least 1, so no traffic flow is silently dropped from the Poisson
    sampling distribution.  The GCD reduction gives the most compact form.
    """
    tm = tm.copy().astype(float)
    np.fill_diagonal(tm, 0.0)

    nonzero_mask = tm > 0
    if not nonzero_mask.any():
        return tm.astype(int)

    min_val = float(tm[nonzero_mask].min())
    scaled = np.round(tm / min_val).astype(int)
    np.fill_diagonal(scaled, 0)

    common = reduce(gcd, scaled[scaled > 0].tolist())
    if common > 1:
        scaled = scaled // common
        np.fill_diagonal(scaled, 0)

    return scaled
