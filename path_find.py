import igraph
import os
import numpy as np
import torch
from tqdm import tqdm

DATA_DIR = "./data/processed/"
# Relations: 63
# Edges: 35,649,195
# Entities: 5,864,272

def load_data(path):
    with open(path, "r") as f:
        data = f.readlines()
    return [s.strip().replace("biolink:", "") for s in data]


class Data:
    def __init__(self, data_dir):
        edges_path = data_dir + "edges.bin"

        self.edges = np.memmap(edges_path, dtype=np.int32, mode="r").reshape(-1, 3)

        self.entities_map = load_data(data_dir + "entities.txt")
        self.relations_map = load_data(data_dir + "relations.txt")

    def __len__(self):
        return self.edges.shape[0]



def get_paths(src, dst, max_len):
    kg = Data(DATA_DIR)

    graph = igraph.Graph(n=len(kg.entities_map))
    graph.add_edges(kg.edges[:, [0,2]])

    # contains vertices along the path
    paths = graph.get_all_simple_paths(src, to=dst, maxlen=max_len)

    # get relations between each vertex
    relations = []
    for path in paths:
        relation =[]
        for i in range(len(path) - 1):
            mask1 = (kg.edges[:,0] == path[i]) & (kg.edges[:,2] == path[i+1])
            mask2 = (kg.edges[:,2] == path[i]) & (kg.edges[:,0] == path[i+1])
            final_mask = (mask1 | mask2)
            relation.append(kg.edges[final_mask,1])
        relations.append(relation)

    # return the node paths and relations
    return paths, relations

def main():
    paths, relations = get_paths(4025261, 5673721, 4)
    print(paths)
    print(relations)


if __name__ == "__main__":
    main()

