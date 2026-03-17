import numpy as np
import pandas as pd
import igraph
import torch


DATA_DIR = "./data/processed/"
# Relations: 63
# Edges: 35,649,195
# Entities: 5,864,272

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.readlines()
    return [s.strip().replace("biolink:", "") for s in data]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, relationship=None):
        edges_path = data_dir + "edges.bin"

        self.edges = np.memmap(edges_path, dtype=np.int32, mode="r").reshape(-1, 3)

        self.entities_map = load_data(data_dir + "entities.txt")
        self.relations_map = load_data(data_dir + "relations.txt")
        if relationship is not None:
            val = self.relations_map.index(relationship)
            self.edges = self.edges[np.where(self.edges[:,1] == val)]

    def __len__(self):
        return self.edges.shape[0]
    
    def __getitem__(self, idx):
        edge = torch.Tensor(np.array(self.edges[idx])).to(torch.long)
        return edge


def get_paths(src, dst, max_len, graph):
    """
    finds paths between two nodes of a certain max length
    paths returned will be (max_len) and (max_len-1) len long
    if the path was shorter than max_len, -1 will be returned for unused indices
    """
    # contains vertices along the path
    paths = graph.get_all_simple_paths(src, to=dst, maxlen=max_len, mode="ALL")

    # get relations between each vertex
    vertices = []
    relations = []
    for path in paths:
        # skip single length path
        if len(path) == 2:
            continue
        vertex = [-1]*(max_len+1)
        relation = [-1]*(max_len)
        for i in range(len(path)-1):
            vertex[i] = path[i]
            eid = graph.get_eid(path[i], path[i+1], directed=False, error=False)
            if eid != -1:
                relation[i] = (graph.es[eid]["relation"])
        vertex[len(path) - 1] = path[len(path) - 1]
        vertices.append(vertex)
        relations.append(relation)

    # return the node paths and relations
    return vertices, relations


def main():
    kg = Dataset(DATA_DIR)

    graph = igraph.Graph(n=len(kg.entities_map))
    graph.add_edges(kg.edges[:, [0,2]])
    graph.es["relation"] = kg.edges[:,1]

    pos_dataset = Dataset(DATA_DIR, 'treats')
    neg_dataset = Dataset(DATA_DIR, 'contraindicated_in')
    vertices = []
    relations = []
    n_pos = len(pos_dataset)
    n_neg = len(neg_dataset)
    i=0

    # find paths between drug that treats disease
    for v1, e, v2 in pos_dataset:
        v, r = get_paths(v1, v2, 3, graph)
        vertices += v
        relations += r
        i+=1
        if i % 5 == 0:
            print(f"On vertex pair {i} out of {n_pos}, current number of paths is {len(vertices)}")
        if (len(vertices)) > 10_000_000:
            break

    print("Dataset has ", len(vertices), " entries")
    print("writing positive datasets to disk")
    node_paths = np.array(vertices, dtype=np.int32)      # shape (N,4)
    relation_paths = np.array(relations, dtype=np.int32)  # shape (N,3)
    node_paths.tofile(DATA_DIR + "pos_path_nodes.bin")
    relation_paths.tofile(DATA_DIR + "pos_path_relations.bin")

    # now do it for negative dataset
    vertices = []
    relations = []
    i=0

    for v1, e, v2 in neg_dataset:
        v, r = get_paths(v1, v2, 3, graph)
        vertices += v
        relations += r
        i+=1
        if i % 5 == 0:
            print(f"On vertex pair {i} out of {n_neg}, current number of paths is {len(vertices)}")
        if (len(vertices)) > 10_000_000:
            break

    print("Dataset has ", len(vertices), " entries")
    print("writing negative datasets to disk")
    node_paths = np.array(vertices, dtype=np.int32)      # shape (N,4)
    relation_paths = np.array(relations, dtype=np.int32)  # shape (N,3)
    node_paths.tofile(DATA_DIR + "neg_path_nodes.bin")
    relation_paths.tofile(DATA_DIR + "neg_path_relations.bin")

if __name__ == "__main__":
    main()

