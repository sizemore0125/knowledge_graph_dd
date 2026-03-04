import json
import struct
from pathlib import Path

import bmt
import igraph as ig

DATA_DIR = "/home/logansizemore/Documents/knowledge_graph_dd/data/"
NODES_FILE = DATA_DIR + "kg2.10.3-conflated-nodes.jsonl"
EDGES_FILE = DATA_DIR + "kg2.10.3-conflated-edges.jsonl"

OUT_DIR = Path(DATA_DIR + "processed")
OUT_DIR.mkdir(exist_ok=True)

ENTITIES_PATH = OUT_DIR / "entities.txt"
RELATIONS_PATH = OUT_DIR / "relations.txt"
EDGES_BIN_PATH = OUT_DIR / "edges.bin"
SUBCLASS_EDGE_LIST_PATH = OUT_DIR / "subclass_edge_list.txt"
RELATION_HIERARCHY_EDGE_LIST_PATH = OUT_DIR / "relation_hierarchy_edge_list.txt"

SKIP_RELATIONS = {
    "biolink:close_match",
    "biolink:same_as",
}
EXACT_MATCH_RELATION = "biolink:exact_match"
SUBCLASS_RELATION = "biolink:subclass_of"
PUBLICATION_CATEGORIES = {"biolink:Publication", "Publication"}


def normalize_name(name, fallback):
    text = name or fallback
    return text.replace("\t", " ").replace("\n", " ").strip() or fallback


def build_entity_index(nodes_file):
    entity2id = {}
    entity_names = []
    publication_node_ids = set()
    node_categories = {}

    with open(nodes_file, "r") as f:
        for line in f:
            node = json.loads(line)
            node_id = node["id"]

            if node_id not in entity2id:
                entity2id[node_id] = len(entity2id)
                entity_names.append(normalize_name(node.get("name"), node_id))

            raw_categories = node.get("category") or []
            if isinstance(raw_categories, str):
                raw_categories = [raw_categories]

            node_categories[node_id] = tuple(raw_categories)
            if any(category in PUBLICATION_CATEGORIES for category in raw_categories):
                publication_node_ids.add(node_id)

    return entity2id, entity_names, publication_node_ids, node_categories


def find_root(parent, idx):
    while parent[idx] != idx:
        parent[idx] = parent[parent[idx]]
        idx = parent[idx]
    return idx


def union(parent, rank, left, right):
    left_root = find_root(parent, left)
    right_root = find_root(parent, right)

    if left_root == right_root:
        return False

    if rank[left_root] < rank[right_root]:
        left_root, right_root = right_root, left_root

    parent[right_root] = left_root
    if rank[left_root] == rank[right_root]:
        rank[left_root] += 1

    return True


def collapse_exact_matches(edges_file, entity2id):
    parent = list(range(len(entity2id)))
    rank = [0] * len(entity2id)
    merge_count = 0

    with open(edges_file, "r") as fin:
        for line in fin:
            edge = json.loads(line)
            if edge["predicate"] != EXACT_MATCH_RELATION:
                continue

            head_idx = entity2id.get(edge["subject"])
            tail_idx = entity2id.get(edge["object"])
            if head_idx is None or tail_idx is None:
                continue

            if union(parent, rank, head_idx, tail_idx):
                merge_count += 1

    return parent, merge_count


def write_entities(entity2id, entity_names, parent):
    component_sizes = {}
    for idx in range(len(entity_names)):
        root = find_root(parent, idx)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    merged_aliases = {}
    for node_id, raw_idx in entity2id.items():
        root = find_root(parent, raw_idx)
        if component_sizes[root] > 1:
            merged_aliases.setdefault(root, []).append(entity_names[raw_idx])

    collapsed_entities = 0
    root_to_entity_id = {}

    with open(ENTITIES_PATH, "w") as fout:
        for node_id, raw_idx in entity2id.items():
            root = find_root(parent, raw_idx)
            entity_id = root_to_entity_id.get(root)
            if entity_id is None:
                entity_id = len(root_to_entity_id)
                root_to_entity_id[root] = entity_id
                if component_sizes[root] > 1:
                    fout.write("\t".join(merged_aliases[root]) + "\n")
                    collapsed_entities += 1
                else:
                    fout.write(entity_names[raw_idx] + "\n")

            entity2id[node_id] = entity_id

    return len(root_to_entity_id), collapsed_entities


def prune_scc_internal_edges(edge_pairs):
    unique_edges = sorted(set(edge_pairs))
    if not unique_edges:
        return [], 0, True

    node_ids = {node_id for edge in unique_edges for node_id in edge}
    local_node2id = {node_id: idx for idx, node_id in enumerate(node_ids)}
    local_edges = [(local_node2id[left], local_node2id[right]) for left, right in unique_edges]

    graph = ig.Graph(
        n=len(local_node2id),
        edges=local_edges,
        directed=True,
    )
    components = graph.connected_components(mode="STRONG")

    vertex_to_component = {}
    cyclic_components = set()
    for component_id, component in enumerate(components):
        if len(component) > 1:
            cyclic_components.add(component_id)
        for vertex_id in component:
            vertex_to_component[vertex_id] = component_id

    pruned_edges = []
    removed_edge_count = 0
    for left, right in unique_edges:
        left_component = vertex_to_component[local_node2id[left]]
        right_component = vertex_to_component[local_node2id[right]]
        if left_component == right_component and left_component in cyclic_components:
            removed_edge_count += 1
            continue
        pruned_edges.append((left, right))

    if not pruned_edges:
        return [], removed_edge_count, True

    pruned_node_ids = {node_id for edge in pruned_edges for node_id in edge}
    pruned_local_node2id = {node_id: idx for idx, node_id in enumerate(pruned_node_ids)}
    pruned_local_edges = [
        (pruned_local_node2id[left], pruned_local_node2id[right])
        for left, right in pruned_edges
    ]
    is_dag = ig.Graph(
        n=len(pruned_local_node2id),
        edges=pruned_local_edges,
        directed=True,
    ).is_dag()

    return pruned_edges, removed_edge_count, is_dag


def stream_edges(edges_file, entity2id, publication_entity_ids):
    relation2id = {}
    edge_count = 0
    raw_subclass_edges = []
    skipped_publication_subclass_edges = 0

    with open(edges_file, "r") as fin, open(EDGES_BIN_PATH, "wb") as fout:
        for line in fin:
            edge = json.loads(line)

            head = edge["subject"]
            relation = edge["predicate"]
            tail = edge["object"]

            if relation in SKIP_RELATIONS or relation == EXACT_MATCH_RELATION:
                continue
            if head not in entity2id or tail not in entity2id:
                continue

            head_id = entity2id[head]
            tail_id = entity2id[tail]
            if head_id == tail_id:
                continue

            if relation == SUBCLASS_RELATION:
                if head_id in publication_entity_ids or tail_id in publication_entity_ids:
                    skipped_publication_subclass_edges += 1
                    continue
                raw_subclass_edges.append((head_id, tail_id))
                continue

            if relation not in relation2id:
                relation2id[relation] = len(relation2id)

            relation_id = relation2id[relation]
            fout.write(struct.pack("iii", head_id, relation_id, tail_id))
            edge_count += 1

            if edge_count % 1_000_000 == 0:
                print(f"{edge_count:,} edges written")

    return relation2id, edge_count, raw_subclass_edges, skipped_publication_subclass_edges


def build_and_write_subclass_graph(
    raw_subclass_edges,
    entity2id,
    publication_entity_ids,
    node_categories,
    base_entity_count,
    toolkit,
):
    pruned_node_edges, removed_node_scc_edges, node_graph_is_dag = prune_scc_internal_edges(raw_subclass_edges)

    category_ancestors = {}
    all_category_labels = set()
    for node_id, categories in node_categories.items():
        entity_id = entity2id.get(node_id)
        if entity_id is None or entity_id in publication_entity_ids:
            continue

        for category in categories:
            if category not in category_ancestors:
                ancestors = toolkit.get_ancestors(category, formatted=True) or [category]
                category_ancestors[category] = tuple(ancestors)
                all_category_labels.update(ancestors)

    category_labels = sorted(all_category_labels)
    category2id = {}
    with open(ENTITIES_PATH, "a") as fout:
        for offset, category_label in enumerate(category_labels):
            category_id = base_entity_count + offset
            category2id[category_label] = category_id
            fout.write(f"{category_label}\n")

    extra_edges = set()
    for node_id, categories in node_categories.items():
        entity_id = entity2id.get(node_id)
        if entity_id is None or entity_id in publication_entity_ids:
            continue

        for category in categories:
            category_id = category2id.get(category)
            if category_id is not None and entity_id != category_id:
                extra_edges.add((entity_id, category_id))

            ancestors = category_ancestors.get(category, (category,))
            for child_category, parent_category in zip(ancestors, ancestors[1:]):
                child_id = category2id[child_category]
                parent_id = category2id[parent_category]
                if child_id != parent_id:
                    extra_edges.add((child_id, parent_id))

    final_edges, removed_augmented_scc_edges, final_is_dag = prune_scc_internal_edges(
        list(pruned_node_edges) + list(extra_edges)
    )

    with open(SUBCLASS_EDGE_LIST_PATH, "w") as fout:
        for head_id, tail_id in final_edges:
            fout.write(f"{head_id}\t{tail_id}\n")

    return {
        "category_nodes": len(category2id),
        "node_edges_before_pruning": len(raw_subclass_edges),
        "node_edges_after_pruning": len(pruned_node_edges),
        "removed_node_scc_edges": removed_node_scc_edges,
        "node_graph_is_dag": node_graph_is_dag,
        "node_to_category_edges": sum(1 for edge in extra_edges if edge[0] < base_entity_count and edge[1] >= base_entity_count),
        "category_to_category_edges": sum(1 for edge in extra_edges if edge[0] >= base_entity_count and edge[1] >= base_entity_count),
        "removed_augmented_scc_edges": removed_augmented_scc_edges,
        "final_edges": len(final_edges),
        "final_is_dag": final_is_dag,
    }


def write_relation_hierarchy_edge_list(relation2id, toolkit):
    hierarchy_edges = set()

    for relation_name in relation2id:
        ancestors = toolkit.get_ancestors(relation_name, formatted=True) or [relation_name]
        for child_name, parent_name in zip(ancestors, ancestors[1:]):
            child_id = relation2id.get(child_name)
            parent_id = relation2id.get(parent_name)
            if child_id is None or parent_id is None or child_id == parent_id:
                continue
            hierarchy_edges.add((child_id, parent_id))

    with open(RELATION_HIERARCHY_EDGE_LIST_PATH, "w") as fout:
        for child_id, parent_id in sorted(hierarchy_edges):
            fout.write(f"{child_id}\t{parent_id}\n")

    return len(hierarchy_edges)


def write_mapping(mapping, path):
    with open(path, "w") as f:
        for key, idx in sorted(mapping.items(), key=lambda item: item[1]):
            f.write(f"{key}\n")


def main():
    toolkit = bmt.Toolkit()

    print("Loading nodes...")
    entity2id, entity_names, publication_node_ids, node_categories = build_entity_index(NODES_FILE)
    print(f"Raw entities: {len(entity2id):,}")
    print(f"Publication nodes: {len(publication_node_ids):,}")

    print("Collapsing exact matches...")
    parent, merge_count = collapse_exact_matches(EDGES_FILE, entity2id)
    base_entity_count, collapsed_entities = write_entities(entity2id, entity_names, parent)
    print(f"Exact-match unions: {merge_count:,}")
    print(f"Collapsed entities: {collapsed_entities:,}")
    print(f"Entities after collapse: {base_entity_count:,}")

    publication_entity_ids = {
        entity2id[node_id]
        for node_id in publication_node_ids
        if node_id in entity2id
    }
    print(f"Publication entities after collapse: {len(publication_entity_ids):,}")

    print("Streaming edges...")
    (
        relation2id,
        edge_count,
        raw_subclass_edges,
        skipped_publication_subclass_edges,
    ) = stream_edges(EDGES_FILE, entity2id, publication_entity_ids)
    print(f"Relations: {len(relation2id):,}")
    print(f"Edges kept: {edge_count:,}")

    print("Building subclass graph...")
    subclass_stats = build_and_write_subclass_graph(
        raw_subclass_edges,
        entity2id,
        publication_entity_ids,
        node_categories,
        base_entity_count,
        toolkit,
    )
    print(f"Category nodes added: {subclass_stats['category_nodes']:,}")
    print(f"Entities after adding categories: {base_entity_count + subclass_stats['category_nodes']:,}")
    print(f"Subclass node-node edges before pruning: {subclass_stats['node_edges_before_pruning']:,}")
    print(f"Subclass edges skipped for publications: {skipped_publication_subclass_edges:,}")
    print(f"Subclass node-node edges removed inside SCCs: {subclass_stats['removed_node_scc_edges']:,}")
    print(f"Subclass node-node edges after pruning: {subclass_stats['node_edges_after_pruning']:,}")
    print(f"Subclass node-node graph is DAG: {subclass_stats['node_graph_is_dag']}")
    print(f"Synthetic node->category edges: {subclass_stats['node_to_category_edges']:,}")
    print(f"Synthetic category->category edges: {subclass_stats['category_to_category_edges']:,}")
    print(f"Subclass edges removed after augmentation SCC pruning: {subclass_stats['removed_augmented_scc_edges']:,}")
    print(f"Subclass edges written: {subclass_stats['final_edges']:,}")
    print(f"Subclass graph is DAG: {subclass_stats['final_is_dag']}")

    print("Writing mappings...")
    write_mapping(relation2id, RELATIONS_PATH)
    hierarchy_edge_count = write_relation_hierarchy_edge_list(relation2id, toolkit)
    print(f"Relation hierarchy edges written: {hierarchy_edge_count:,}")

    print("Done.")


if __name__ == "__main__":
    main()
