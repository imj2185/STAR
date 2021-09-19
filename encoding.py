import torch
from torch_geometric.utils import get_laplacian
from torch_geometric.utils.num_nodes import maybe_num_nodes


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

    def eval(self, edge_index, edge_attr, num_nodes=None):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        edge_index, edge_attr = get_laplacian(edge_index, edge_attr,
                                              normalization=self.normalization,
                                              num_nodes=num_nodes)
        # TODO the second term below is not correct
        return edge_index, torch.exp(-self.beta * edge_attr)
