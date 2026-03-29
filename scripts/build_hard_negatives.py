import dbm
import random
import struct
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory

import bmt
import numpy as np

DATA_DIR = Path("/home/logansizemore/Documents/knowledge_graph_dd/data/processed")
ENTITIES_PATH = DATA_DIR / "entities.txt"
RELATIONS_PATH = DATA_DIR / "relations.txt"
EDGES_BIN_PATH = DATA_DIR / "edges.bin"
SUBCLASS_EDGE_LIST_PATH = DATA_DIR / "subclass_edge_list.txt"
NEG_EDGES_BIN_PATH = DATA_DIR / "negative_edges.bin"

TRIPLE_STRUCT = struct.Struct("iii")
ANTONYM_RELATION_PAIRS = (
    ("biolink:has_part", "biolink:lacks_part"),
    ("biolink:has_increased_amount", "biolink:has_decreased_amount"),
    ("biolink:treats", "biolink:contraindicated_in"),
    ("biolink:promotes_condition", "biolink:preventative_for_condition"),
    ("biolink:exacerbates_condition", "biolink:preventative_for_condition"),
)


def pack_key(h, r, t):
    return TRIPLE_STRUCT.pack(int(h), int(r), int(t))


def load_relations():
    with open(RELATIONS_PATH, "r") as f:
        return [line.strip() for line in f]


def load_entity_category_ids():
    category_ids = set()
    category_name_to_id = {}
    num_entities = 0

    with open(ENTITIES_PATH, "r") as f:
        for entity_id, line in enumerate(f):
            name = line.rstrip("\n")
            if name.startswith("biolink:"):
                category_ids.add(entity_id)
                category_name_to_id[name] = entity_id
            num_entities = entity_id + 1

    return num_entities, category_ids, category_name_to_id


def build_primary_categories(num_entities, category_ids):
    primary_category = array("i", [-1]) * num_entities
    parents_by_child = {}
    children_by_parent = {}

    with open(SUBCLASS_EDGE_LIST_PATH, "r") as f:
        for line in f:
            child_str, parent_str = line.rstrip("\n").split("\t")
            child = int(child_str)
            parent = int(parent_str)

            if child in category_ids:
                continue

            if parent in category_ids:
                if primary_category[child] == -1:
                    primary_category[child] = parent
            else:
                parents_by_child.setdefault(child, array("I")).append(parent)
                children_by_parent.setdefault(parent, array("I")).append(child)

    return primary_category, parents_by_child, children_by_parent


def build_relation_metadata(relations, category_name_to_id, toolkit):
    relation_to_id = {name: idx for idx, name in enumerate(relations)}
    relation_signatures = {}
    relations_by_signature = {}
    allowed_heads = [None] * len(relations)
    allowed_tails = [None] * len(relations)
    antonym_by_relation = {}

    for relation_id, relation_name in enumerate(relations):
        slot = toolkit.get_element(relation_name)
        if slot is None or slot.domain is None or slot.range is None:
            continue

        signature = (slot.domain, slot.range)
        relation_signatures[relation_id] = signature
        relations_by_signature.setdefault(signature, []).append(relation_id)

        head_allowed_names = toolkit.get_descendants(slot.domain, formatted=True) or [slot.domain]
        tail_allowed_names = toolkit.get_descendants(slot.range, formatted=True) or [slot.range]

        head_allowed_ids = {category_name_to_id[name] for name in head_allowed_names if name in category_name_to_id}
        tail_allowed_ids = {category_name_to_id[name] for name in tail_allowed_names if name in category_name_to_id}

        if head_allowed_ids:
            allowed_heads[relation_id] = head_allowed_ids
        if tail_allowed_ids:
            allowed_tails[relation_id] = tail_allowed_ids

    for left_name, right_name in ANTONYM_RELATION_PAIRS:
        left_id = relation_to_id.get(left_name)
        right_id = relation_to_id.get(right_name)
        if left_id is None or right_id is None:
            continue
        antonym_by_relation[left_id] = right_id
        antonym_by_relation[right_id] = left_id

    return relation_signatures, relations_by_signature, allowed_heads, allowed_tails, antonym_by_relation


def is_schema_valid(h, r, t, primary_category, allowed_heads, allowed_tails):
    head_allowed = allowed_heads[r]
    if head_allowed is not None:
        head_category = primary_category[h]
        if head_category == -1 or head_category not in head_allowed:
            return False

    tail_allowed = allowed_tails[r]
    if tail_allowed is not None:
        tail_category = primary_category[t]
        if tail_category == -1 or tail_category not in tail_allowed:
            return False

    return True


def sample_sibling(entity_id, parents_by_child, children_by_parent):
    parents = parents_by_child.get(entity_id)
    if not parents:
        return None

    for _ in range(20):
        parent = parents[random.randrange(len(parents))]
        siblings = children_by_parent.get(parent)
        if not siblings or len(siblings) <= 1:
            continue
        candidate = siblings[random.randrange(len(siblings))]
        if candidate != entity_id:
            return int(candidate)

    return None


def try_insert(key, pos_db, neg_db):
    if key in pos_db or key in neg_db:
        return False
    neg_db[key] = b"1"
    return True


def propose_candidate(
    rule_id,
    edge_index,
    h,
    r,
    t,
    edges,
    relation_edge_indices,
    antonym_by_relation,
    parents_by_child,
    children_by_parent,
    relation_signatures,
    relations_by_signature,
    primary_category,
    allowed_heads,
    allowed_tails,
):
    if rule_id == 1:
        r_antonym = antonym_by_relation.get(r)
        if r_antonym is None:
            return None
        candidate = (h, r_antonym, t)
        if is_schema_valid(candidate[0], candidate[1], candidate[2], primary_category, allowed_heads, allowed_tails):
            return candidate
        return None

    if rule_id == 2:
        replace_head = (edge_index % 2) == 0
        for _ in range(30):
            pool = relation_edge_indices[r]
            if not pool:
                return None
            other_idx = pool[random.randrange(len(pool))]
            if replace_head:
                h_new = int(edges[other_idx, 0])
                if h_new == h or primary_category[h_new] != primary_category[h]:
                    continue
                trial = (h_new, r, t)
            else:
                t_new = int(edges[other_idx, 2])
                if t_new == t or primary_category[t_new] != primary_category[t]:
                    continue
                trial = (h, r, t_new)

            if is_schema_valid(trial[0], trial[1], trial[2], primary_category, allowed_heads, allowed_tails):
                return trial
        return None

    if rule_id == 3:
        replace_head = (edge_index % 2) == 1
        if replace_head:
            h_new = sample_sibling(h, parents_by_child, children_by_parent)
            if h_new is None or primary_category[h_new] != primary_category[h]:
                return None
            trial = (h_new, r, t)
        else:
            t_new = sample_sibling(t, parents_by_child, children_by_parent)
            if t_new is None or primary_category[t_new] != primary_category[t]:
                return None
            trial = (h, r, t_new)

        if is_schema_valid(trial[0], trial[1], trial[2], primary_category, allowed_heads, allowed_tails):
            return trial
        return None

    signature = relation_signatures.get(r)
    if signature is None:
        return None
    swap_candidates = relations_by_signature.get(signature, ())
    if len(swap_candidates) <= 1:
        return None

    for _ in range(20):
        r_new = swap_candidates[random.randrange(len(swap_candidates))]
        if r_new == r:
            continue
        trial = (h, r_new, t)
        if is_schema_valid(trial[0], trial[1], trial[2], primary_category, allowed_heads, allowed_tails):
            return trial

    return None


def generate_hard_negatives(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    relations = load_relations()
    num_relations = len(relations)
    num_entities, category_ids, category_name_to_id = load_entity_category_ids()

    print(f"Loaded {num_entities:,} entities and {num_relations:,} relations")

    primary_category, parents_by_child, children_by_parent = build_primary_categories(
        num_entities=num_entities,
        category_ids=category_ids,
    )

    toolkit = bmt.Toolkit()
    (
        relation_signatures,
        relations_by_signature,
        allowed_heads,
        allowed_tails,
        antonym_by_relation,
    ) = build_relation_metadata(
        relations=relations,
        category_name_to_id=category_name_to_id,
        toolkit=toolkit,
    )

    edges = np.memmap(EDGES_BIN_PATH, dtype=np.int32, mode="r").reshape(-1, 3)
    num_edges = edges.shape[0]
    print(f"Loaded {num_edges:,} positive edges")

    relation_edge_indices = [array("I") for _ in range(num_relations)]

    with TemporaryDirectory(prefix="neg_gen_") as tmpdir:
        pos_db_path = str(Path(tmpdir) / "pos_db")
        neg_db_path = str(Path(tmpdir) / "neg_db")

        with dbm.open(pos_db_path, "n") as pos_db, dbm.open(neg_db_path, "n") as neg_db:
            print("Indexing positive edges...")
            for edge_idx in range(num_edges):
                h, r, t = edges[edge_idx]
                relation_edge_indices[int(r)].append(edge_idx)
                pos_db[pack_key(h, r, t)] = b"1"

            rule_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            print("Generating negatives...")
            negative_count = 0
            first_pass_misses = 0

            with open(NEG_EDGES_BIN_PATH, "wb") as fout:
                for edge_idx in range(num_edges):
                    h = int(edges[edge_idx, 0])
                    r = int(edges[edge_idx, 1])
                    t = int(edges[edge_idx, 2])

                    ordered_rules = [((edge_idx % 4) + 1)]
                    ordered_rules += [rule_id for rule_id in (1, 2, 3, 4) if rule_id != ordered_rules[0]]

                    accepted = None
                    accepted_rule = None

                    for rule_id in ordered_rules:
                        candidate = propose_candidate(
                            rule_id=rule_id,
                            edge_index=edge_idx,
                            h=h,
                            r=r,
                            t=t,
                            edges=edges,
                            relation_edge_indices=relation_edge_indices,
                            antonym_by_relation=antonym_by_relation,
                            parents_by_child=parents_by_child,
                            children_by_parent=children_by_parent,
                            relation_signatures=relation_signatures,
                            relations_by_signature=relations_by_signature,
                            primary_category=primary_category,
                            allowed_heads=allowed_heads,
                            allowed_tails=allowed_tails,
                        )
                        if candidate is None:
                            continue

                        if try_insert(pack_key(candidate[0], candidate[1], candidate[2]), pos_db, neg_db):
                            accepted = candidate
                            accepted_rule = rule_id
                            break

                    if accepted is None:
                        first_pass_misses += 1
                        continue

                    fout.write(TRIPLE_STRUCT.pack(*accepted))
                    rule_counts[accepted_rule] += 1
                    negative_count += 1

                    if negative_count % 1_000_000 == 0:
                        print(f"  {negative_count:,} / {num_edges:,} negatives written")

                refill_attempts = 0
                max_refill_attempts = num_edges * 200
                print("Refilling negatives to match positive count...")

                while negative_count < num_edges:
                    refill_attempts += 1
                    src_idx = random.randrange(num_edges)
                    h = int(edges[src_idx, 0])
                    r = int(edges[src_idx, 1])
                    t = int(edges[src_idx, 2])

                    ordered_rules = [((refill_attempts % 4) + 1)]
                    ordered_rules += [rule_id for rule_id in (1, 2, 3, 4) if rule_id != ordered_rules[0]]

                    accepted = None
                    accepted_rule = None
                    for rule_id in ordered_rules:
                        candidate = propose_candidate(
                            rule_id=rule_id,
                            edge_index=src_idx + refill_attempts,
                            h=h,
                            r=r,
                            t=t,
                            edges=edges,
                            relation_edge_indices=relation_edge_indices,
                            antonym_by_relation=antonym_by_relation,
                            parents_by_child=parents_by_child,
                            children_by_parent=children_by_parent,
                            relation_signatures=relation_signatures,
                            relations_by_signature=relations_by_signature,
                            primary_category=primary_category,
                            allowed_heads=allowed_heads,
                            allowed_tails=allowed_tails,
                        )
                        if candidate is None:
                            continue
                        if try_insert(pack_key(candidate[0], candidate[1], candidate[2]), pos_db, neg_db):
                            accepted = candidate
                            accepted_rule = rule_id
                            break

                    if accepted is None:
                        if refill_attempts >= max_refill_attempts:
                            raise RuntimeError(
                                f"Failed to refill negatives. Built {negative_count:,} of {num_edges:,}"
                            )
                        continue

                    fout.write(TRIPLE_STRUCT.pack(*accepted))
                    rule_counts[accepted_rule] += 1
                    negative_count += 1

                    if negative_count % 1_000_000 == 0:
                        print(f"  {negative_count:,} / {num_edges:,} negatives written")

    print(f"Done. Wrote {negative_count:,} negatives to {NEG_EDGES_BIN_PATH}")
    print(f"First-pass misses: {first_pass_misses:,}")
    print(f"Rule 1 count: {rule_counts[1]:,}")
    print(f"Rule 2 count: {rule_counts[2]:,}")
    print(f"Rule 3 count: {rule_counts[3]:,}")
    print(f"Rule 4 count: {rule_counts[4]:,}")


if __name__ == "__main__":
    generate_hard_negatives(seed=0)
