import json
import struct
from pathlib import Path

DATA_DIR = "./data/"
NODES_FILE = DATA_DIR + "kg2.10.3-conflated-nodes.jsonl"
EDGES_FILE = DATA_DIR + "kg2.10.3-conflated-edges.jsonl"

OUT_DIR = Path(DATA_DIR + "processed")
OUT_DIR.mkdir(exist_ok=True)

ENTITIES_PATH = OUT_DIR / "entities.txt"
RELATIONS_PATH = OUT_DIR / "relations.txt"
EDGES_BIN_PATH = OUT_DIR / "edges.bin"

EXCLUDE_RELATIONS = {
    "biolink:close_match",
    "biolink:same_as",
}

def build_entity_index(nodes_file):
    entity2id = {}
    with open(nodes_file, "r") as f:
        for line in f:
            node = json.loads(line)
            eid = node["id"]
            if eid not in entity2id:
                entity2id[eid] = len(entity2id)

    return entity2id


def stream_edges(edges_file, entity2id):
    relation2id = {}
    edge_count = 0

    with open(edges_file, "r") as fin, open(EDGES_BIN_PATH, "wb") as fout:

        for line in fin:
            edge = json.loads(line)

            h = edge["subject"]
            r = edge["predicate"]
            t = edge["object"]

            if r in EXCLUDE_RELATIONS:
                continue

            if h not in entity2id or t not in entity2id:
                continue

            if r not in relation2id:
                relation2id[r] = len(relation2id)

            h_id = entity2id[h]
            r_id = relation2id[r]
            t_id = entity2id[t]

            fout.write(struct.pack("iii", h_id, r_id, t_id))
            edge_count += 1

            if edge_count % 1_000_000 == 0:
                print(f"{edge_count:,} edges written")

    return relation2id, edge_count


def write_mapping(mapping, path):
    with open(path, "w") as f:
        for key, idx in sorted(mapping.items(), key=lambda x: x[1]):
            f.write(f"{key}\n")


def main():
    print("Building entity index...")
    entity2id = build_entity_index(NODES_FILE)
    print(f"Entities: {len(entity2id):,}")

    print("Streaming edges...")
    relation2id, edge_count = stream_edges(EDGES_FILE, entity2id)

    print(f"Relations: {len(relation2id):,}")
    print(f"Edges kept: {edge_count:,}")

    print("Writing mappings...")
    write_mapping(entity2id, ENTITIES_PATH)
    write_mapping(relation2id, RELATIONS_PATH)

    print("Done.")


if __name__ == "__main__":
    main()