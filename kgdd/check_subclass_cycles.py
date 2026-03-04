import json

import igraph as ig

DATA_DIR = "/home/logansizemore/Documents/knowledge_graph_dd/data/"
NODES_FILE = DATA_DIR + "kg2.10.3-conflated-nodes.jsonl"
EDGES_FILE = DATA_DIR + "kg2.10.3-conflated-edges.jsonl"

SUBCLASS_RELATION = "biolink:subclass_of"
PUBLICATION_CATEGORIES = {"biolink:Publication", "Publication"}


def load_entities_and_publications(nodes_file):
    entity2id = {}
    publication_nodes = set()

    with open(nodes_file, "r") as f:
        for line in f:
            node = json.loads(line)
            node_id = node["id"]
            entity2id[node_id] = len(entity2id)

            raw_categories = node.get("category") or []
            if isinstance(raw_categories, str):
                raw_categories = [raw_categories]

            if any(category in PUBLICATION_CATEGORIES for category in raw_categories):
                publication_nodes.add(node_id)

    return entity2id, publication_nodes


def load_subclass_edges(edges_file, entity2id, publication_nodes):
    edge_list = []
    skipped_publication_edges = 0

    with open(edges_file, "r") as f:
        for line in f:
            edge = json.loads(line)
            if edge.get("predicate") != SUBCLASS_RELATION:
                continue

            subject = edge["subject"]
            obj = edge["object"]

            if subject in publication_nodes or obj in publication_nodes:
                skipped_publication_edges += 1
                continue

            if subject not in entity2id or obj not in entity2id:
                continue

            mapped_edge = (entity2id[subject], entity2id[obj])
            edge_list.append(mapped_edge)

    return edge_list, skipped_publication_edges


def remove_scc_internal_edges(edge_list):
    if not edge_list:
        return [], 0

    local_node2id = {}
    unique_local_edges = set()

    for left_global, right_global in edge_list:
        left_local = local_node2id.setdefault(left_global, len(local_node2id))
        right_local = local_node2id.setdefault(right_global, len(local_node2id))
        unique_local_edges.add((left_local, right_local))

    graph = ig.Graph(
        n=len(local_node2id),
        edges=list(unique_local_edges),
        directed=True,
    )
    components = graph.connected_components(mode="STRONG")

    vertex_to_component = {}
    nontrivial_component_ids = set()

    for component_id, component in enumerate(components):
        if len(component) > 1:
            nontrivial_component_ids.add(component_id)
        for vertex_id in component:
            vertex_to_component[vertex_id] = component_id

    pruned_edges = []
    removed_edge_count = 0

    for left_global, right_global in edge_list:
        left_local = local_node2id[left_global]
        right_local = local_node2id[right_global]
        left_component = vertex_to_component[left_local]
        right_component = vertex_to_component[right_local]

        if left_component == right_component and left_component in nontrivial_component_ids:
            removed_edge_count += 1
            continue

        pruned_edges.append((left_global, right_global))

    return pruned_edges, removed_edge_count


def summarize_component_sizes(component_sizes):
    if not component_sizes:
        return "[]"

    if len(component_sizes) <= 20:
        return str(component_sizes)

    return f"{component_sizes[:20]} ... ({len(component_sizes):,} total)"


def main():
    print("Loading entities...")
    entity2id, publication_nodes = load_entities_and_publications(NODES_FILE)
    print(f"Entities: {len(entity2id):,}")
    print(f"Publication nodes: {len(publication_nodes):,}")

    print("Loading subclass_of edges...")
    edge_list, skipped_publication_edges = load_subclass_edges(
        EDGES_FILE,
        entity2id,
        publication_nodes,
    )
    filtered_edge_set = set(edge_list)
    filtered_nodes = {node_id for edge in filtered_edge_set for node_id in edge}

    pruned_edge_list, removed_scc_edge_count = remove_scc_internal_edges(edge_list)
    pruned_edge_set = set(pruned_edge_list)
    pruned_nodes = {node_id for edge in pruned_edge_set for node_id in edge}
    removed_node_count = len(filtered_nodes - pruned_nodes)

    pruned_node2id = {node_id: idx for idx, node_id in enumerate(pruned_nodes)}
    pruned_local_edges = [
        (pruned_node2id[left], pruned_node2id[right])
        for left, right in pruned_edge_set
    ]

    pruned_graph = ig.Graph(
        n=len(pruned_nodes),
        edges=pruned_local_edges,
        directed=True,
    )
    is_dag = pruned_graph.is_dag()
    components = pruned_graph.connected_components(mode="STRONG")
    component_sizes = sorted(
        (len(component) for component in components if len(component) > 1),
        reverse=True,
    )

    print("whole-subclass-graph")
    print(f"  subclass_of relations: {len(pruned_edge_list):,}")
    print(f"  entities in subgraph: {len(pruned_nodes):,}")
    print(f"  unique edges in subgraph: {len(pruned_edge_set):,}")
    print(f"  is_dag: {is_dag}")
    print(f"  nontrivial strongly connected components: {len(component_sizes):,}")
    print(f"  component sizes: {summarize_component_sizes(component_sizes)}")
    print(f"  removed nodes: {removed_node_count:,}")
    print(f"  skipped publication edges: {skipped_publication_edges:,}")
    print(f"  removed SCC-internal edges: {removed_scc_edge_count:,}")


if __name__ == "__main__":
    main()
