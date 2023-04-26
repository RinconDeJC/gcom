import numpy as np

class UnionFind:

    def __init__(self, s):
        self.s = s
        self.n_components = s
        self.tree_size = np.ones((s,))
        self.parent = np.arange(0, s)

    def node_union(self, p: int, q: int):
        if p < 0 or q < 0 or p >= self.s or q >= self.s:
            raise ValueError(f'Incorrect arguments out of domain 0..{self.s}')
        i = self.find(p)
        j = self.find(q)
        if (i == j):
            return
        self.n_components -= 1
        if (self.tree_size[i] < self.tree_size[j]):
            self.parent[i] = self.parent[j]
            self.tree_size[j] += self.tree_size[i]
        else:
            self.parent[j] = self.parent[i]
            self.tree_size[i] += self.tree_size[j]
    
    def find(self, p: int) -> int:
        root = p
        to_fix = []
        while self.parent[root] != root:
            to_fix.append(root)
            root = self.parent[root]
        # Path compression
        for q in to_fix[::-1]:
            if self.parent[q] != root:
                self.tree_size[self.parent[q]] -= self.tree_size[q]
                self.parent[q] = root
        return root
    
    def size(self, p: int) -> int:
        i = self.find(p)
        return self.tree_size[i]

    def components(self) -> int:
        return self.n_components

    def connected(self, p: int, q: int) -> bool:
        return self.find(p) == self.find(q)