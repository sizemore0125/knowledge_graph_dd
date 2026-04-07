import os

import igraph as ig
import numpy as np
import torch
from tqdm import tqdm
import skeletonkey as sk

from kgdd.data import build_relation_swap_split, load_data, load_edges, load_hierarchy
from kgdd.dataset import DomainRangeDataset, NegativeDataset, PositiveDataset, RelationSwapTestDataset
from kgdd.disk_loss import DiskLoss
from kgdd.model import Model
from kgdd.rule_penalties import RulePenalties
from kgdd.training import test_binary_classification_metrics


@sk.unlock("configs/config.yaml")
def main(args):
    entities_path = args.data_dir + "entities.txt"
    relations_path = args.data_dir + "relations.txt"

    entities_hierarchy_edge_list_path = args.data_dir + "subclass_edge_list.txt"
    relations_hierarchy_edge_list_path = args.data_dir + "relation_hierarchy_edge_list.txt"

    knowledge_graph_edges_path = args.data_dir + "edges.bin"
    negative_edges_path = args.data_dir + "negative_edges.bin"

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(1, os.cpu_count() or 1)
    pin_memory = device.type == "cuda"

    entities_map = load_data(entities_path)
    relations_map = load_data(relations_path)
    num_entities = len(entities_map)
    num_relations = len(relations_map)

    entity_hierarchy_edges = load_hierarchy(entities_hierarchy_edge_list_path)
    relation_hierarchy_edges = load_hierarchy(relations_hierarchy_edge_list_path)

    entity_hierarchy = ig.Graph(n=num_entities, edges=entity_hierarchy_edges, directed=True)
    entity_hierarchy.vs["name"] = entities_map

    relation_hierarchy = ig.Graph(n=num_relations, edges=relation_hierarchy_edges, directed=True)
    relation_hierarchy.vs["name"] = relations_map

    positive_edges = load_edges(knowledge_graph_edges_path)
    negative_edges = load_edges(negative_edges_path)
    train_positive_edges, train_negative_edges, relation_swap_test_edges = build_relation_swap_split(
        pos_edges=positive_edges,
        neg_edges=negative_edges,
        relations_map=relations_map,
        num_entities=num_entities,
        num_relations=num_relations,
    )

    pos_dataset = PositiveDataset(
        edges=train_positive_edges,
        relation_hierarchy=relation_hierarchy,
    )
    neg_dataset = NegativeDataset(edges=train_negative_edges)

    split_ratio = 0.01
    pos_dataset_train_len, pos_dataset_test_len = (
        len(pos_dataset) - int(len(pos_dataset) * split_ratio),
        int(len(pos_dataset) * split_ratio),
    )
    neg_dataset_train_len, neg_dataset_test_len = (
        len(neg_dataset) - int(len(neg_dataset) * split_ratio),
        int(len(neg_dataset) * split_ratio),
    )

    train_pos_dataset, test_pos_dataset = torch.utils.data.random_split(pos_dataset, [pos_dataset_train_len, pos_dataset_test_len])
    train_neg_dataset, test_neg_dataset = torch.utils.data.random_split(neg_dataset, [neg_dataset_train_len, neg_dataset_test_len])

    train_pos_indices = np.asarray(train_pos_dataset.indices, dtype=np.int64)
    train_pos_edges = np.asarray(train_positive_edges[train_pos_indices], dtype=np.int32)

    domain_range_dataset = DomainRangeDataset(
        edges=train_pos_edges,
        entity_hierarchy=entity_hierarchy,
        relation_hierarchy=relation_hierarchy,
        num_samples=len(train_pos_dataset),
    )
    dataset = torch.utils.data.ConcatDataset([train_pos_dataset, train_neg_dataset, domain_range_dataset])
    test_dataset = torch.utils.data.ConcatDataset([test_pos_dataset, test_neg_dataset])
    relation_swap_test_dataset = RelationSwapTestDataset(
        true_edges=relation_swap_test_edges,
        relations_map=relations_map,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    relation_swap_test_dataloader = torch.utils.data.DataLoader(
        dataset=relation_swap_test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = Model(
        n_entities=num_entities,
        n_relations=num_relations,
        entity_hierarchy=entity_hierarchy,
        relation_hierarchy=relation_hierarchy,
        random_inverse_edges=True,
    ).to(device)

    model.train()

    rule_penalties = RulePenalties(model)
    disk_loss = DiskLoss(model.entity_codebook, entity_hierarchy).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    metrics_pbar = tqdm(total=0, position=1, bar_format="{desc}", leave=False)

    try:
        for epoch_idx in range(args.epochs):
            epoch_desc = f"Epoch {epoch_idx + 1}/{args.epochs}"
            with tqdm(
                train_dataloader,
                total=len(train_dataloader),
                miniters=50,
                mininterval=1,
                desc=epoch_desc,
                position=0,
            ) as pbar:
                for step_idx, (edges, aux_edges, labels, task_ids) in enumerate(pbar):
                    edges = edges.to(device, non_blocking=pin_memory)
                    aux_edges = aux_edges.to(device, non_blocking=pin_memory)
                    labels = labels.to(device, non_blocking=pin_memory).unsqueeze(1)
                    task_ids = task_ids.to(device, non_blocking=pin_memory)

                    logits = model(edges)
                    base_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

                    kg_pos_mask = (task_ids == 0) & (labels.squeeze(1) > 0.5)
                    if kg_pos_mask.any():
                        pos_scores = logits[kg_pos_mask]
                        general_edges = aux_edges[kg_pos_mask]
                        general_scores = model(general_edges)
                        pos_edges = edges[kg_pos_mask]

                        entity_ids = torch.cat([pos_edges[:, 0], pos_edges[:, 2]], dim=0)
                        negative_ids = entity_ids[torch.randperm(entity_ids.shape[0], device=device)]

                        hierarchy_penalty = torch.nn.functional.relu(pos_scores - general_scores).squeeze(1).mean()
                        disk_penalty = disk_loss(entity_ids, negative_ids)
                    else:
                        hierarchy_penalty = torch.tensor(0.0, device=device)
                        disk_penalty = torch.tensor(0.0, device=device)

                    domain_range_mask = task_ids == 1
                    if domain_range_mask.any():
                        rule_penalty = rule_penalties.domain_range_loss(
                            edges=edges[domain_range_mask],
                            true_edges=aux_edges[domain_range_mask],
                            logits=logits[domain_range_mask],
                        )
                    else:
                        rule_penalty = torch.tensor(0.0, device=device)

                    loss = base_loss + 0.25 * hierarchy_penalty + 0.25 * rule_penalty + 0.25 * disk_penalty

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (step_idx + 1) % 50 == 0:
                        pbar.set_postfix(
                            loss=f"{loss.item():.4f}",
                            rel_hier=f"{hierarchy_penalty.item():.4f}",
                            disk=f"{disk_penalty.item():.4f}",
                            rule=f"{rule_penalty.item():.4f}",
                            refresh=False,
                        )

                    if step_idx % 5000 == 0:
                        checkpoint_path = os.path.join(args.checkpoint_dir, "model.pt")
                        metrics = test_binary_classification_metrics(model, test_dataloader, device)
                        relation_swap_metrics = test_binary_classification_metrics(
                            model, relation_swap_test_dataloader, device
                        )
                        metrics_pbar.set_description_str(
                            f"test acc={metrics['accuracy']:.4f} p={metrics['precision']:.4f} r={metrics['recall']:.4f} f1={metrics['f1']:.4f} | "
                            f"swap acc={relation_swap_metrics['accuracy']:.4f} p={relation_swap_metrics['precision']:.4f} r={relation_swap_metrics['recall']:.4f} f1={relation_swap_metrics['f1']:.4f}"
                        )
                        metrics_pbar.refresh()
                        torch.save(model.state_dict(), checkpoint_path)
    finally:
        metrics_pbar.close()


if __name__ == "__main__":
    main()
