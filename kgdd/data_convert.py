import json
import struct
from pathlib import Path

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
RELATION_HIERARCHY = [
    ("related_to", "has_member"),
    ("related_to", "in_taxon"),
    ("related_to", "located_in"),
    ("located_in", "occurs_in"),
    ("located_in", "disease_has_location"),
    ("related_to", "disease_has_basis_in"),
    ("related_to", "has_part"),
    ("has_part", "has_plasma_membrane_part"),
    ("has_part", "composed_primarily_of"),
    ("has_part", "lacks_part"),
    ("related_to", "develops_from"),
    ("develops_from", "derives_from"),
    ("related_to", "overlaps"),
    ("related_to", "associated_with"),
    ("associated_with", "correlated_with"),
    ("associated_with", "coexists_with"),
    ("associated_with", "temporally_related_to"),
    ("related_to", "interacts_with"),
    ("interacts_with", "physically_interacts_with"),
    ("physically_interacts_with", "directly_physically_interacts_with"),
    ("physically_interacts_with", "indirectly_physically_interacts_with"),
    ("interacts_with", "colocalizes_with"),
    ("related_to", "affects"),
    ("affects", "causes"),
    ("affects", "contributes_to"),
    ("affects", "exacerbates_condition"),
    ("affects", "predisposes_to_condition"),
    ("affects", "disrupts"),
    ("related_to", "regulates"),
    ("related_to", "treats"),
    ("treats", "applied_to_treat"),
    ("treats", "treats_or_applied_or_studied_to_treat"),
    ("treats", "preventative_for_condition"),
    ("treats", "in_clinical_trials_for"),
    ("treats", "contraindicated_in"),
    ("related_to", "diagnoses"),
    ("diagnoses", "biomarker_for"),
    ("related_to", "gene_associated_with_condition"),
    ("related_to", "gene_product_of"),
    ("related_to", "expressed_in"),
    ("related_to", "is_sequence_variant_of"),
    ("is_sequence_variant_of", "has_molecular_consequence"),
    ("related_to", "produces"),
    ("produces", "has_metabolite"),
    ("related_to", "has_input"),
    ("related_to", "has_output"),
    ("related_to", "has_participant"),
    ("related_to", "actively_involved_in"),
    ("related_to", "enables"),
    ("related_to", "capable_of"),
    ("related_to", "has_increased_amount"),
    ("related_to", "has_decreased_amount"),
    ("related_to", "has_not_completed"),
    ("related_to", "precedes"),
    ("related_to", "chemically_similar_to"),
    ("related_to", "homologous_to"),
    ("related_to", "broad_match"),
    ("related_to", "drug_regulatory_status_world_wide"),
    ("related_to", "manifestation_of"),
    ("related_to", "has_phenotype"),
]


def normalize_name(name, fallback):
    text = name or fallback
    return text.replace("\t", " ").replace("\n", " ").strip() or fallback


def build_entity_index(nodes_file):
    entity2id = {}
    entity_names = []
    with open(nodes_file, "r") as f:
        for line in f:
            node = json.loads(line)
            eid = node["id"]
            if eid not in entity2id:
                entity2id[eid] = len(entity2id)
                entity_names.append(normalize_name(node.get("name"), eid))

    return entity2id, entity_names


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


def stream_edges(edges_file, entity2id):
    relation2id = {}
    edge_count = 0
    subclass_edge_count = 0

    with (
        open(edges_file, "r") as fin,
        open(EDGES_BIN_PATH, "wb") as fout,
        open(SUBCLASS_EDGE_LIST_PATH, "w") as subclass_out,
    ):

        for line in fin:
            edge = json.loads(line)

            h = edge["subject"]
            r = edge["predicate"]
            t = edge["object"]

            if r in SKIP_RELATIONS or r == EXACT_MATCH_RELATION:
                continue

            if h not in entity2id or t not in entity2id:
                continue

            h_id = entity2id[h]
            t_id = entity2id[t]

            if h_id == t_id:
                continue

            if r == SUBCLASS_RELATION:
                subclass_out.write(f"{h_id}\t{t_id}\n")
                subclass_edge_count += 1
                continue

            if r not in relation2id:
                relation2id[r] = len(relation2id)

            r_id = relation2id[r]

            fout.write(struct.pack("iii", h_id, r_id, t_id))
            edge_count += 1

            if edge_count % 1_000_000 == 0:
                print(f"{edge_count:,} edges written")

    return relation2id, edge_count, subclass_edge_count


def write_relation_hierarchy_edge_list(relation2id):
    edge_count = 0
    skipped_edges = 0

    with open(RELATION_HIERARCHY_EDGE_LIST_PATH, "w") as fout:
        for parent_name, child_name in RELATION_HIERARCHY:
            parent_id = relation2id.get(f"biolink:{parent_name}")
            child_id = relation2id.get(f"biolink:{child_name}")

            if parent_id is None or child_id is None:
                skipped_edges += 1
                continue

            fout.write(f"{parent_id}\t{child_id}\n")
            edge_count += 1

    return edge_count, skipped_edges


def write_mapping(mapping, path):
    with open(path, "w") as f:
        for key, idx in sorted(mapping.items(), key=lambda x: x[1]):
            f.write(f"{key}\n")


def main():
    print("Relations Edges:", len(RELATION_HIERARCHY))
    print("Building entity index...")
    entity2id, entity_names = build_entity_index(NODES_FILE)
    print(f"Raw entities: {len(entity2id):,}")

    print("Collapsing exact matches...")
    parent, merge_count = collapse_exact_matches(EDGES_FILE, entity2id)
    total_entities, collapsed_entities = write_entities(entity2id, entity_names, parent)
    print(f"Exact-match unions: {merge_count:,}")
    print(f"Collapsed entities: {collapsed_entities:,}")
    print(f"Entities after collapse: {total_entities:,}")

    print("Streaming edges...")
    relation2id, edge_count, subclass_edge_count = stream_edges(EDGES_FILE, entity2id)

    print(f"Relations: {len(relation2id):,}")
    print(f"Edges kept: {edge_count:,}")
    print(f"Subclass edges written: {subclass_edge_count:,}")

    print("Writing mappings...")
    write_mapping(relation2id, RELATIONS_PATH)
    hierarchy_edge_count, skipped_hierarchy_edges = write_relation_hierarchy_edge_list(relation2id)
    print(f"Relation hierarchy edges written: {hierarchy_edge_count:,}")
    print(f"Relation hierarchy edges skipped: {skipped_hierarchy_edges:,}")

    print("Done.")


if __name__ == "__main__":
    main()
