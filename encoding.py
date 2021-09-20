import torch
from torch_geometric.utils import get_laplacian
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import spspmm


class PositionalEncoding(object):
    def __init__(self, zero_diagonal=False) -> None:
        super().__init__()
        self.zero_diagonal = zero_diagonal
        self.cached_pos_enc = None

    def eval(self, graph, **kwargs):
        pass


class DiffusionEncoding(PositionalEncoding):
    def __init__(self,
                 beta=1., 
                 use_edge_attr=False,
                 normalization=None,
                 zero_diagonal=False) -> None:
        super().__init__(zero_diagonal=zero_diagonal)
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def eval(self, graph, num_nodes=None):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        num_nodes = maybe_num_nodes(graph.edge_index, num_nodes)
        edge_index, edge_attr = get_laplacian(graph.edge_index, edge_attr,
                                              normalization=self.normalization,
                                              num_nodes=num_nodes)
        # TODO the second term below seems not correct
        return edge_index, torch.exp(-self.beta * edge_attr)


class KStepRandomWalkEncoding(PositionalEncoding):
    def __init__(self,
                 p=3,
                 beta=0.5,
                 use_edge_attr=False,
                 normalization=None,
                 zero_diagonal=False):
        super().__init__(zero_diagonal=zero_diagonal)
        self.p = p
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def eval(self, graph, num_nodes=None):
        # graph:
        num_nodes = maybe_num_nodes(graph.edge_index, num_nodes)
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(graph.edge_index, edge_attr,
                                              normalization=self.normalization,
                                              num_nodes=num_nodes)
        ei, ea = edge_index, edge_attr
        for _ in range(self.p - 1):
            ei, ea = spspmm(ei, ea, edge_index, edge_attr, num_nodes, num_nodes, num_nodes)
        return ei, ea

