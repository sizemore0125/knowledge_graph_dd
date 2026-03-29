import numpy as np


def load_data(path):
    with open(path, "r") as f:
        data = f.readlines()
    return [s.strip().replace("biolink:", "") for s in data]


def load_hierarchy(path):
    edges = []

    with open(path, "r") as f:
        for line in f:
            child_str, parent_str = line.rstrip("\n").split("\t")
            child_id = int(child_str)
            parent_id = int(parent_str)
            edges.append((child_id, parent_id))

    return edges


def load_edges(path):
    return np.memmap(path, dtype=np.int32, mode="r").reshape(-1, 3)


def edge_keys(edges, num_entities, num_relations):
    heads = edges[:, 0].astype(np.uint64, copy=False)
    rels = edges[:, 1].astype(np.uint64, copy=False)
    tails = edges[:, 2].astype(np.uint64, copy=False)
    return heads + np.uint64(num_entities) * (rels + np.uint64(num_relations) * tails)


def build_relation_swap_split(pos_edges, neg_edges, relations_map, num_entities, num_relations):
    relation_to_id = {name: idx for idx, name in enumerate(relations_map)}
    treats_id = relation_to_id.get("treats")
    contra_id = relation_to_id.get("contraindicated_in")
    if treats_id is None or contra_id is None:
        raise ValueError("relations.txt must include both 'treats' and 'contraindicated_in'.")

    treats_idx = np.where(pos_edges[:, 1] == treats_id)[0]
    contra_idx = np.where(pos_edges[:, 1] == contra_id)[0]

    heldout_treats_idx = treats_idx[np.random.rand(len(treats_idx)) > 0.5]
    heldout_contra_idx = contra_idx[np.random.rand(len(contra_idx)) > 0.5]

    heldout_idx = np.concatenate([heldout_treats_idx, heldout_contra_idx], axis=0)
    test_edges = np.asarray(pos_edges[heldout_idx], dtype=np.int64)

    train_pos_mask = np.ones(len(pos_edges), dtype=bool)
    train_pos_mask[heldout_idx] = False
    train_pos_edges = np.asarray(pos_edges[train_pos_mask], dtype=np.int32)

    keep_neg_mask = np.ones(len(neg_edges), dtype=bool)

    relevant_neg_idx = np.where((neg_edges[:, 1] == treats_id) | (neg_edges[:, 1] == contra_id))[0]
    if len(relevant_neg_idx) > 0 and len(test_edges) > 0:
        relevant_neg_edges = np.asarray(neg_edges[relevant_neg_idx], dtype=np.int64)
        relevant_neg_keys = edge_keys(relevant_neg_edges, num_entities=num_entities, num_relations=num_relations)
        test_keys = edge_keys(test_edges, num_entities=num_entities, num_relations=num_relations)
        overlap_mask = np.isin(relevant_neg_keys, test_keys)
        keep_neg_mask[relevant_neg_idx[overlap_mask]] = False

    train_neg_edges = np.asarray(neg_edges[keep_neg_mask], dtype=np.int32)

    return train_pos_edges, train_neg_edges, test_edges
