from array import array
from collections import deque
from pathlib import Path

DATA_DIR = Path("/home/logansizemore/Documents/knowledge_graph_dd/data/processed")
ENTITIES_PATH = DATA_DIR / "entities.txt"
SUBCLASS_EDGE_LIST_PATH = DATA_DIR / "subclass_edge_list.txt"

DEFAULT_EMBEDDING_DIM = 8


def count_lines(path):
    with open(path, "r") as f:
        return sum(1 for _ in f)


def load_graph(edge_path, num_nodes):
    in_degree = array("I", [0]) * num_nodes
    out_degree = array("I", [0]) * num_nodes
    touched = bytearray(num_nodes)
    touched_nodes = 0
    edge_count = 0

    with open(edge_path, "r") as f:
        for line in f:
            left_str, right_str = line.rstrip("\n").split("\t")
            left = int(left_str)
            right = int(right_str)

            out_degree[left] += 1
            in_degree[right] += 1

            if not touched[left]:
                touched[left] = 1
                touched_nodes += 1
            if not touched[right]:
                touched[right] = 1
                touched_nodes += 1

            edge_count += 1

    offsets = array("Q", [0]) * (num_nodes + 1)
    running = 0
    for node_id in range(num_nodes):
        offsets[node_id] = running
        running += out_degree[node_id]
    offsets[num_nodes] = running

    parents = array("I", [0]) * edge_count
    cursor = array("Q", offsets[:-1])

    with open(edge_path, "r") as f:
        for line in f:
            left_str, right_str = line.rstrip("\n").split("\t")
            left = int(left_str)
            right = int(right_str)

            write_idx = cursor[left]
            parents[write_idx] = right
            cursor[left] += 1

    return {
        "edge_count": edge_count,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "offsets": offsets,
        "parents": parents,
        "touched": touched,
        "touched_nodes": touched_nodes,
    }


def topo_sort(num_nodes, in_degree, offsets, parents):
    work_in_degree = array("I", in_degree)
    queue = deque(node_id for node_id in range(num_nodes) if work_in_degree[node_id] == 0)
    topo_order = array("I")

    while queue:
        node_id = queue.popleft()
        topo_order.append(node_id)

        start = offsets[node_id]
        stop = offsets[node_id + 1]
        for edge_idx in range(start, stop):
            parent_id = parents[edge_idx]
            work_in_degree[parent_id] -= 1
            if work_in_degree[parent_id] == 0:
                queue.append(parent_id)

    return topo_order


def compute_depths(num_nodes, topo_order, offsets, parents):
    depths = array("I", [0]) * num_nodes

    for node_id in reversed(topo_order):
        start = offsets[node_id]
        stop = offsets[node_id + 1]
        best = 0
        for edge_idx in range(start, stop):
            parent_id = parents[edge_idx]
            candidate = depths[parent_id] + 1
            if candidate > best:
                best = candidate
        depths[node_id] = best

    return depths


def summarize_depths(num_nodes, touched, in_degree, out_degree, depths):
    active_depth_sum = 0
    active_nodes = 0
    roots = 0
    leaves = 0
    isolated = 0
    max_depth = 0
    max_depth_nodes = 0
    leaf_depth_sum = 0
    leaf_max_depth = 0

    for node_id in range(num_nodes):
        if not touched[node_id]:
            continue

        active_nodes += 1
        depth = depths[node_id]
        active_depth_sum += depth

        if depth > max_depth:
            max_depth = depth
            max_depth_nodes = 1
        elif depth == max_depth:
            max_depth_nodes += 1

        node_in = in_degree[node_id]
        node_out = out_degree[node_id]

        if node_out == 0 and node_in > 0:
            roots += 1
        elif node_in == 0 and node_out > 0:
            leaves += 1
            leaf_depth_sum += depth
            if depth > leaf_max_depth:
                leaf_max_depth = depth
        elif node_in == 0 and node_out == 0:
            isolated += 1

    avg_depth = active_depth_sum / active_nodes if active_nodes else 0.0
    avg_leaf_depth = leaf_depth_sum / leaves if leaves else 0.0

    return {
        "active_nodes": active_nodes,
        "roots": roots,
        "leaves": leaves,
        "isolated": isolated,
        "max_depth": max_depth,
        "max_depth_nodes": max_depth_nodes,
        "avg_depth": avg_depth,
        "avg_leaf_depth": avg_leaf_depth,
        "leaf_max_depth": leaf_max_depth,
    }


def estimate_embedding_cost(num_entities, active_nodes, edge_count, embedding_dim):
    bytes_per_matrix = num_entities * embedding_dim * 4
    bytes_per_active_slice = active_nodes * embedding_dim * 4

    return {
        "embedding_dim": embedding_dim,
        "parameter_bytes": bytes_per_matrix,
        "clone_bytes_per_forward": bytes_per_matrix,
        "active_slice_bytes": bytes_per_active_slice,
        "vector_adds_per_forward": edge_count,
        "scalar_adds_per_forward": edge_count * embedding_dim,
    }


def format_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def main():
    print("Counting entities...")
    num_entities = count_lines(ENTITIES_PATH)
    print(f"Total entities: {num_entities:,}")

    print("Loading subclass graph...")
    graph = load_graph(SUBCLASS_EDGE_LIST_PATH, num_entities)
    print(f"Subclass edges: {graph['edge_count']:,}")
    print(f"Entities touched by subclass graph: {graph['touched_nodes']:,}")
    print(f"Entities outside subclass graph: {num_entities - graph['touched_nodes']:,}")

    print("Topological sort...")
    topo_order = topo_sort(
        num_entities,
        graph["in_degree"],
        graph["offsets"],
        graph["parents"],
    )
    is_dag = len(topo_order) == num_entities
    print(f"Is DAG: {is_dag}")

    if not is_dag:
        print(f"Nodes emitted in topological order: {len(topo_order):,}")
        print("The current HierarchicalEmbedding will fail on this graph.")
        return

    print("Computing longest-path depths...")
    depths = compute_depths(
        num_entities,
        topo_order,
        graph["offsets"],
        graph["parents"],
    )
    depth_stats = summarize_depths(
        num_entities,
        graph["touched"],
        graph["in_degree"],
        graph["out_degree"],
        depths,
    )

    print("Graph shape")
    print(f"  roots (out-degree 0): {depth_stats['roots']:,}")
    print(f"  leaves (in-degree 0): {depth_stats['leaves']:,}")
    print(f"  isolated touched nodes: {depth_stats['isolated']:,}")
    print(f"  average node depth: {depth_stats['avg_depth']:.2f}")
    print(f"  max node depth: {depth_stats['max_depth']:,}")
    print(f"  nodes at max depth: {depth_stats['max_depth_nodes']:,}")
    print(f"  average leaf-to-root path length: {depth_stats['avg_leaf_depth']:.2f}")
    print(f"  max leaf-to-root path length: {depth_stats['leaf_max_depth']:,}")

    cost = estimate_embedding_cost(
        num_entities=num_entities,
        active_nodes=graph["touched_nodes"],
        edge_count=graph["edge_count"],
        embedding_dim=DEFAULT_EMBEDDING_DIM,
    )

    print("Embedding cost estimate")
    print(f"  embedding dim: {cost['embedding_dim']}")
    print(f"  residual parameter matrix: {format_bytes(cost['parameter_bytes'])}")
    print(f"  full clone per forward: {format_bytes(cost['clone_bytes_per_forward'])}")
    print(f"  active subgraph slice size: {format_bytes(cost['active_slice_bytes'])}")
    print(f"  vector additions per forward: {cost['vector_adds_per_forward']:,}")
    print(f"  scalar additions per forward: {cost['scalar_adds_per_forward']:,}")
    print("  note: the current HierarchicalEmbedding recomputes the full matrix every forward,")
    print("        so this work is done even when only a small batch of node IDs is requested.")


if __name__ == "__main__":
    main()
