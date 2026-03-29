import bmt
import numpy as np
import torch
from tqdm import tqdm


class DomainRangeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        edges: np.ndarray,
        entity_hierarchy,
        relation_hierarchy,
        num_samples: int,
    ) -> None:
        self.edges = edges
        self.entity_hierarchy = entity_hierarchy
        self.relation_hierarchy = relation_hierarchy
        self.toolkit = bmt.Toolkit()
        self.num_samples = num_samples

        self.relation_specs = self.build_relation_specs()
        if len(self.relation_specs) == 0:
            raise ValueError("DomainRangeDataset has no valid relation specs to sample from.")

    def normalize_name(self, name: str) -> str:
        return name.replace("biolink:", "")

    def get_toolkit_element(self, name: str):
        return self.toolkit.get_element(name) or self.toolkit.get_element(f"biolink:{name}")

    def expand_category(self, category_name: str) -> np.ndarray:
        node_ids = self.entity_hierarchy.subcomponent(category_name, mode="IN")
        return np.asarray(node_ids, dtype=np.int64)

    def build_relation_specs(self) -> list[dict]:
        # Characterize the relations and store information about each relation.
        specs = []
        all_entities = np.arange(self.entity_hierarchy.vcount(), dtype=np.int64)

        for relation_id, relation_name in tqdm(enumerate(self.relation_hierarchy.vs["name"]), desc="Characterizing DR Dataset:"):
            # Get name of domain and range entity.
            relation_slot = self.get_toolkit_element(relation_name)

            if relation_slot is None:
                continue

            if relation_slot.domain is None or relation_slot.range is None:
                continue

            domain_element = self.get_toolkit_element(relation_slot.domain)
            range_element = self.get_toolkit_element(relation_slot.range)

            if domain_element is None or range_element is None:
                continue

            domain_name = self.normalize_name(domain_element.class_uri)
            range_name = self.normalize_name(range_element.class_uri)

            # Get entity id within the entity hierarchy.
            try:
                domain_category_id = self.entity_hierarchy.vs.find(name=domain_name).index
                range_category_id = self.entity_hierarchy.vs.find(name=range_name).index
            except ValueError:
                continue

            # Get enetities rooted at the subtree of the domain/range.
            domain_pool = self.expand_category(domain_name)
            range_pool = self.expand_category(range_name)

            if len(domain_pool) == 0 or len(range_pool) == 0:
                continue

            # Get entities that do not match the domain/range.
            out_domain_pool = all_entities[~np.isin(all_entities, domain_pool)]
            out_range_pool = all_entities[~np.isin(all_entities, range_pool)]

            if len(out_domain_pool) == 0 or len(out_range_pool) == 0:
                continue

            # Get all edges that have this relation.
            relation_mask = self.edges[:, 1] == relation_id
            graph_edges = np.asarray(self.edges[relation_mask], dtype=np.int64)

            if len(graph_edges) == 0:
                continue

            # Get of all edges that have this relation as a set.
            graph_edge_set = {(int(head_id), int(rel_id), int(tail_id)) for head_id, rel_id, tail_id in graph_edges}

            specs.append(
                {
                    "relation_id": relation_id,
                    "domain_pool": domain_pool,
                    "range_pool": range_pool,
                    "out_domain_pool": out_domain_pool,
                    "out_range_pool": out_range_pool,
                    "graph_edges": graph_edges,
                    "graph_edge_set": graph_edge_set,
                    "true_edge": np.asarray([domain_category_id, relation_id, range_category_id], dtype=np.int64),
                }
            )

        return specs

    def sample_in_graph(self, spec: dict) -> tuple[np.ndarray, float]:
        edge = spec["graph_edges"][np.random.randint(len(spec["graph_edges"]))]
        return edge, 1.0

    def sample_rule_valid_not_in_graph(self, spec: dict) -> tuple[np.ndarray, float]:
        relation_id = spec["relation_id"]
        for _ in range(100):
            head_id = int(spec["domain_pool"][np.random.randint(len(spec["domain_pool"]))])
            tail_id = int(spec["range_pool"][np.random.randint(len(spec["range_pool"]))])
            edge = (head_id, relation_id, tail_id)
            if edge not in spec["graph_edge_set"]:
                return np.asarray(edge, dtype=np.int64), 0.0
        return self.sample_rule_breaking_not_in_graph(spec)

    def sample_rule_breaking_not_in_graph(self, spec: dict) -> tuple[np.ndarray, float]:
        relation_id = spec["relation_id"]
        for _ in range(100):
            case_id = np.random.randint(3)
            if case_id == 0:
                head_id = int(spec["out_domain_pool"][np.random.randint(len(spec["out_domain_pool"]))])
                tail_id = int(spec["range_pool"][np.random.randint(len(spec["range_pool"]))])
            elif case_id == 1:
                head_id = int(spec["domain_pool"][np.random.randint(len(spec["domain_pool"]))])
                tail_id = int(spec["out_range_pool"][np.random.randint(len(spec["out_range_pool"]))])
            else:
                head_id = int(spec["out_domain_pool"][np.random.randint(len(spec["out_domain_pool"]))])
                tail_id = int(spec["out_range_pool"][np.random.randint(len(spec["out_range_pool"]))])

            edge = (head_id, relation_id, tail_id)
            if edge not in spec["graph_edge_set"]:
                return np.asarray(edge, dtype=np.int64), 0.0

        head_id = int(spec["out_domain_pool"][np.random.randint(len(spec["out_domain_pool"]))])
        tail_id = int(spec["out_range_pool"][np.random.randint(len(spec["out_range_pool"]))])
        return np.asarray([head_id, relation_id, tail_id], dtype=np.int64), 0.0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = self.relation_specs[np.random.randint(len(self.relation_specs))]
        draw = np.random.rand()

        if draw < 0.5:
            edge_np, label = self.sample_in_graph(spec)
        elif draw < 0.75:
            edge_np, label = self.sample_rule_valid_not_in_graph(spec)
        else:
            edge_np, label = self.sample_rule_breaking_not_in_graph(spec)

        edge = torch.tensor(edge_np, dtype=torch.long)
        true_edge = torch.tensor(spec["true_edge"], dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float32)
        task_id = torch.tensor(1, dtype=torch.long)
        return edge, true_edge, label, task_id
