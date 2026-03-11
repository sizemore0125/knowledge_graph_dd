"""
Recurrent Message Passing Graph Neural Network trained for relation detection between drug and disease (specifically relation 'cures')
Using an initial embedding of the drug and disease in a 'vertex space' which is enhanced by knowledge of existing paths 
between the drug and disease of interest, which is then predicted whether a drug and disease should have an edge with 'cures'
between them. 

Written for final project for Stephen Ramsey's Biological Networks class
by Cameron Hatler and Logan Sizemore
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualPathStep(nn.Module):
    """
    Model that learns residual weights along a path
    from a source to a target by gathering information
    of nodes and relations along path
    """
    
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, node_i, rel_i, node_j):
        """
        node_i: [batch, emb_dim]
        rel_i:  [batch, emb_dim]
        node_j: [batch, emb_dim]
        """
        
        x = torch.cat([node_i, rel_i, node_j], dim=-1)
        residual = self.net(x)
        
        return residual

class PathReasoningModel(nn.Module):
    """
    Class used to find residuals along an entire path
    between two vertices, work is done by ResidualPathStep
    model for each step along the path
    """
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        
        self.step = ResidualPathStep(emb_dim, hidden_dim)
        
    def forward(self, node_embeddings, relation_embeddings):
        """
        node_embeddings: [batch, path_len+1, emb_dim]
        relation_embeddings: [batch, path_len, emb_dim]
        """
        
        h = node_embeddings[:,0]  # starting node
        
        for i in range(relation_embeddings.shape[1]):
            
            r = relation_embeddings[:, i]
            next_node = node_embeddings[:, i+1]
            
            residual = self.step(h, r, next_node)
            
            h = next_node + residual
        # returns the final node and learned residuals from path
        return h
    
class EdgePredictor(nn.Module):
    """
    The model that takes the final embedding
    to determine if there should be an edge
    between the two nodes
    """
    def __init__(self, emb_dim):
        super().__init__()
        
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, 1)
        )
    
    def forward(self, h_path, target_node):
        
        x = torch.cat([h_path, target_node], dim=-1)
        score = self.scorer(x)
        
        return torch.sigmoid(score)
    
class KnowledgeGraphPathModel(nn.Module):
    """
    Putting the path reasoning and Edge Prediction together
    to predict if two nodes should be related based on a 
    different existing path between them
    """
    def __init__(self, emb_dim=8, hidden_dim=16):
        """
        :param emd_dim: the dimension of vertex/relation embedding
        :param hiddien_dim: dimension of hidden layers in the NN
        """
        super().__init__()
        
        self.path_model = PathReasoningModel(emb_dim, hidden_dim)
        self.edge_predictor = EdgePredictor(emb_dim)
        
    def forward(self, nodes, relations, target_node):
        """
        :param nodes: list of node embeddings along path
        :param relations: list of relations along path
        :param target_node: the node we are checking relation to
        """
        
        path_embedding = self.path_model(nodes, relations)
        
        score = self.edge_predictor(path_embedding, target_node)
        
        return score