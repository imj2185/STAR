import queue
import torch

class TreeNode: 
    def __init__(self, value, parent=None, children=None):
        self.value = value
        self.parent = parent
        self.children = children if children else []
        self._positional_encoding = []
        if self.parent is None:
            self.branch = 0
        else:
            if self not in self.parent.children:
                self.parent.children.append(self)
            self.branch = self.parent.children.index(self)
    
    def num_children(self):
        return len(self.children)

    def set_parent(self, parent):
        self.parent = parent

    def set_children(self, children):
        self.children = children if children else []

    def set_branch(self, branch):
        self.branch = branch

    """def get_positional_encoding(self):
        if self._positional_encoding is None:
            if self.parent:
                self._positional_encoding = [
                    0.0 for _ in range(self.parent.num_children())]
                self._positional_encoding[self.branch] = 1.0
                self._positional_encoding += self.parent.get_positional_encoding()
            else:
                self._positional_encoding = []
        return self._positional_encoding"""


def get_padded_positional_encoding(node, max_pos_len):
        padded = node._positional_encoding
        while len(padded) < max_pos_len:
            padded.append(0)
        padded = padded[: max_pos_len]
        return padded

def tree_encoding_from_traversal(onehot_length=7, device='cuda', max_padding=None):
    adj_list = {0:(16, 1, 12),
                1:(20),
                2:(3),
                4:(5),
                5:(6),
                6:(7),
                7:(21, 22),
                8:(9),
                9:(10),
                10:(11),
                11:(23, 24),
                12:(13),
                13:(14),
                14:(15),
                15:None,
                16:(17),
                17:(18),
                18:(19),
                19:None,
                20:(8, 2, 4),
                21:None,
                22:None,
                23:None,
                24:None
                }
    
    tree = {k:TreeNode(None) for k in range(25)}
    for key, tup in adj_list.items():
        curr_node = tree[key]
        curr_node.set_children(tup)
        if isinstance(tup, tuple):
            for i, j in enumerate(tup):
                child_node = tree[j]
                child_node.set_parent(key)
                if len(tup) == 2:
                    child_node.set_branch(i*(onehot_length-1))
                elif len(tup) == 3 and onehot_length == 3:
                    child_node.set_branch(i)
                else:
                    child_node.set_branch(i * (onehot_length-1) // 2)
        else:
            if tup:
                child_node = tree[tup]
                child_node.set_parent(key)
                child_node.set_branch(onehot_length // 2)
        
        tree[key] = curr_node

    q = queue.Queue()
    c = [0 for _ in range(25)]
    c[0] = 1
    q.put(0)
    tree[0]._positional_encoding = [0] * onehot_length
    while(not q.empty()):
        parent = q.get()
        if isinstance(tree[parent].children, tuple): # 2 children or 3 children
            for i, j in enumerate(tree[parent].children):
                next_node = j
                if not c[next_node]:
                    c[next_node] = 1
                    tree[next_node]._positional_encoding = [0] * onehot_length
                    tree[next_node]._positional_encoding[tree[next_node].branch] = 1
                    tree[next_node]._positional_encoding += tree[parent]._positional_encoding
                    q.put(next_node)
        else:
            if tree[parent].children:
                next_node = tree[parent].children
                c[next_node] = 1
                tree[next_node]._positional_encoding = [0] * onehot_length
                tree[next_node]._positional_encoding[tree[next_node].branch] = 1
                tree[next_node]._positional_encoding += tree[parent]._positional_encoding
                q.put(next_node)

    for i in range(len(tree)):
        tree[i]._positional_encoding = get_padded_positional_encoding(tree[i], max_padding)

    return torch.Tensor([tree[i]._positional_encoding for i in range(len(tree))]).to(device)
