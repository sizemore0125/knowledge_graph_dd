import argparse
import random
from pathlib import Path

import bmt
import numpy as np


RELATION_NAMES = [
    "subclass_of",
    "related_to",
    "has_member",
    "has_part",
    "coexists_with",
    "manifestation_of",
    "has_phenotype",
    "located_in",
    "affects",
    "in_taxon",
    "biomarker_for",
    "has_input",
    "gene_associated_with_condition",
    "has_participant",
    "gene_product_of",
    "causes",
    "derives_from",
    "associated_with",
    "capable_of",
    "regulates",
    "produces",
    "physically_interacts_with",
    "expressed_in",
    "correlated_with",
    "chemically_similar_to",
    "has_metabolite",
    "develops_from",
    "occurs_in",
    "has_output",
    "drug_regulatory_status_world_wide",
    "precedes",
    "actively_involved_in",
    "is_sequence_variant_of",
    "has_molecular_consequence",
    "lacks_part",
    "has_plasma_membrane_part",
    "has_increased_amount",
    "overlaps",
    "has_decreased_amount",
    "has_not_completed",
    "temporally_related_to",
    "directly_physically_interacts_with",
    "composed_primarily_of",
    "homologous_to",
    "indirectly_physically_interacts_with",
    "disrupts",
    "contributes_to",
    "exact_match",
    "treats",
    "disease_has_location",
    "disease_has_basis_in",
    "broad_match",
    "treats_or_applied_or_studied_to_treat",
    "interacts_with",
    "predisposes_to_condition",
    "preventative_for_condition",
    "exacerbates_condition",
    "diagnoses",
    "colocalizes_with",
    "enables",
    "contraindicated_in",
    "applied_to_treat",
    "in_clinical_trials_for",
]


FALLBACK_CATEGORY_NAMES = [
    "NamedThing",
    "DiseaseOrPhenotypicFeature",
    "Disease",
    "PhenotypicFeature",
    "ChemicalEntity",
    "Drug",
    "SmallMolecule",
    "Gene",
    "Protein",
    "GeneProduct",
    "SequenceVariant",
    "BiologicalProcess",
    "MolecularActivity",
    "Pathway",
    "AnatomicalEntity",
    "Cell",
    "OrganismTaxon",
    "Publication",
    "ClinicalTrial",
    "GeneFamily",
]


def repo_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def normalize_biolink_name(name: str) -> str:
    return name.replace("biolink:", "")


def collect_category_names() -> list[str]:
    toolkit = bmt.Toolkit()
    category_names = set(FALLBACK_CATEGORY_NAMES)

    for relation_name in RELATION_NAMES:
        relation_slot = toolkit.get_element(relation_name) or toolkit.get_element(f"biolink:{relation_name}")
        if relation_slot is None:
            continue
        if relation_slot.domain is not None:
            category_names.add(normalize_biolink_name(relation_slot.domain))
        if relation_slot.range is not None:
            category_names.add(normalize_biolink_name(relation_slot.range))

    ordered_names = []
    for name in FALLBACK_CATEGORY_NAMES:
        if name in category_names:
            ordered_names.append(name)
            category_names.remove(name)

    ordered_names.extend(sorted(category_names))
    return ordered_names


def build_entity_names(num_entities: int, category_names: list[str]) -> list[str]:
    if num_entities < len(category_names):
        raise ValueError(f"num_entities must be at least {len(category_names)}.")

    names = list(category_names)
    for entity_id in range(len(category_names), num_entities):
        names.append(f"DebugEntity{entity_id:04d}")
    return names


def build_entity_hierarchy(num_entities: int, category_names: list[str], rng: random.Random) -> list[tuple[int, int]]:
    category_count = len(category_names)
    category_to_id = {name: idx for idx, name in enumerate(category_names)}
    root_id = category_to_id.get("NamedThing", 0)
    edges = set()

    parent_by_name = {
        "DiseaseOrPhenotypicFeature": "NamedThing",
        "Disease": "DiseaseOrPhenotypicFeature",
        "PhenotypicFeature": "DiseaseOrPhenotypicFeature",
        "ChemicalEntity": "NamedThing",
        "Drug": "ChemicalEntity",
        "SmallMolecule": "ChemicalEntity",
        "Gene": "NamedThing",
        "Protein": "GeneProduct",
        "GeneProduct": "NamedThing",
        "GeneFamily": "Gene",
        "SequenceVariant": "NamedThing",
        "BiologicalProcess": "NamedThing",
        "MolecularActivity": "BiologicalProcess",
        "Pathway": "BiologicalProcess",
        "AnatomicalEntity": "NamedThing",
        "Cell": "AnatomicalEntity",
        "ClinicalTrial": "Publication",
        "Publication": "NamedThing",
        "OrganismTaxon": "NamedThing",
    }

    for child_name, child_id in category_to_id.items():
        if child_id == root_id:
            continue

        parent_name = parent_by_name.get(child_name)
        parent_id = category_to_id.get(parent_name) if parent_name is not None else None

        if parent_id is None:
            if child_id <= root_id:
                parent_id = root_id
            else:
                parent_id = rng.randrange(root_id + 1) if root_id > 0 else 0

        if child_id != parent_id:
            if child_id > parent_id:
                edges.add((child_id, parent_id))
            else:
                edges.add((child_id, root_id))

    for entity_id in range(category_count, num_entities):
        if entity_id % 11 == 0:
            parent_id = rng.randrange(category_count, entity_id)
        else:
            parent_id = rng.randrange(1, category_count)
        edges.add((entity_id, parent_id))

    return sorted(edges)


def build_relation_hierarchy(num_relations: int, rng: random.Random) -> list[tuple[int, int]]:
    edges = set()

    for relation_id in range(1, min(12, num_relations)):
        parent_id = rng.randrange(relation_id)
        edges.add((relation_id, parent_id))

    manual_edges = [
        ("treats", "treats_or_applied_or_studied_to_treat"),
        ("applied_to_treat", "treats_or_applied_or_studied_to_treat"),
        ("in_clinical_trials_for", "treats_or_applied_or_studied_to_treat"),
        ("directly_physically_interacts_with", "physically_interacts_with"),
        ("indirectly_physically_interacts_with", "physically_interacts_with"),
        ("exact_match", "broad_match"),
        ("has_plasma_membrane_part", "has_part"),
    ]

    relation_to_id = {name: idx for idx, name in enumerate(RELATION_NAMES[:num_relations])}
    for child_name, parent_name in manual_edges:
        child_id = relation_to_id.get(child_name)
        parent_id = relation_to_id.get(parent_name)
        if child_id is not None and parent_id is not None and child_id != parent_id:
            if child_id > parent_id:
                edges.add((child_id, parent_id))
            else:
                edges.add((parent_id, child_id))

    return sorted(edges)


def choose_relation_ids(num_relations: int, num_edges: int, rng: random.Random) -> list[int]:
    relation_ids = list(range(num_relations))
    while len(relation_ids) < num_edges:
        relation_ids.append(rng.randrange(num_relations))
    rng.shuffle(relation_ids)
    return relation_ids[:num_edges]


def build_positive_edges(num_entities: int, num_relations: int, num_edges: int, rng: random.Random) -> np.ndarray:
    relation_ids = choose_relation_ids(num_relations=num_relations, num_edges=num_edges, rng=rng)
    edges = set()

    for relation_id in relation_ids:
        while True:
            head_id = rng.randrange(num_entities)
            tail_id = rng.randrange(num_entities)
            edge = (head_id, relation_id, tail_id)
            if edge not in edges:
                edges.add(edge)
                break

    while len(edges) < num_edges:
        edge = (
            rng.randrange(num_entities),
            rng.randrange(num_relations),
            rng.randrange(num_entities),
        )
        edges.add(edge)

    return np.asarray(sorted(edges), dtype=np.int32)


def build_negative_edges(
    num_entities: int,
    num_relations: int,
    num_edges: int,
    positive_edges: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    positive_edge_set = {tuple(int(x) for x in edge) for edge in positive_edges}
    negative_edges = set()

    while len(negative_edges) < num_edges:
        edge = (
            rng.randrange(num_entities),
            rng.randrange(num_relations),
            rng.randrange(num_entities),
        )
        if edge in positive_edge_set or edge in negative_edges:
            continue
        negative_edges.add(edge)

    return np.asarray(sorted(negative_edges), dtype=np.int32)


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def write_edge_list(path: Path, edges: list[tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for child_id, parent_id in edges:
            f.write(f"{child_id}\t{parent_id}\n")


def write_binary_edges(path: Path, edges: np.ndarray) -> None:
    edges.astype(np.int32, copy=False).tofile(path)


def build_debug_dataset(
    output_dir: Path,
    num_entities: int = 500,
    num_positive_edges: int = 5000,
    num_negative_edges: int = 5000,
    seed: int = 7,
) -> None:
    rng = random.Random(seed)
    num_relations = len(RELATION_NAMES)
    category_names = collect_category_names()

    output_dir.mkdir(parents=True, exist_ok=True)

    entity_names = build_entity_names(num_entities=num_entities, category_names=category_names)
    relation_names = list(RELATION_NAMES)
    entity_hierarchy = build_entity_hierarchy(num_entities=num_entities, category_names=category_names, rng=rng)
    relation_hierarchy = build_relation_hierarchy(num_relations=num_relations, rng=rng)
    positive_edges = build_positive_edges(
        num_entities=num_entities,
        num_relations=num_relations,
        num_edges=num_positive_edges,
        rng=rng,
    )
    negative_edges = build_negative_edges(
        num_entities=num_entities,
        num_relations=num_relations,
        num_edges=num_negative_edges,
        positive_edges=positive_edges,
        rng=rng,
    )

    write_lines(output_dir / "entities.txt", entity_names)
    write_lines(output_dir / "relations.txt", relation_names)
    write_edge_list(output_dir / "subclass_edge_list.txt", entity_hierarchy)
    write_edge_list(output_dir / "relation_hierarchy_edge_list.txt", relation_hierarchy)
    write_binary_edges(output_dir / "edges.bin", positive_edges)
    write_binary_edges(output_dir / "negative_edges.bin", negative_edges)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small random debug knowledge graph dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_data_dir() / "debug_dataset",
        help="Directory to write the debug dataset into.",
    )
    parser.add_argument("--num-entities", type=int, default=500)
    parser.add_argument("--num-positive-edges", type=int, default=5000)
    parser.add_argument("--num-negative-edges", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_debug_dataset(
        output_dir=args.output_dir,
        num_entities=args.num_entities,
        num_positive_edges=args.num_positive_edges,
        num_negative_edges=args.num_negative_edges,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
