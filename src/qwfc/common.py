from abc import abstractmethod

import numpy as np


class AlphabetUser:
    def __init__(self, n_values):
        self._n_values = n_values

    @property
    def n_values(self):
        return self._n_values


class Tile(AlphabetUser):
    def __init__(self, n_values):
        super().__init__(n_values)

    @property
    def n_qubits(self):
        return int(np.ceil(np.log2(self.n_values)))


class TileMap:
    def __init__(self):
        self._tiles = {}
        #
        self._adj = None  # adj[coord][n_key] = n_coord : if present, alpha_{ d=n_key, i=coord, j=n_coord } = 1
        self._adj_offmap = None  # adj_offmap[coord][n_key]: if present, beta_{ d=n_key, i=coord } = 1

    def __getitem__(self, coord):
        return self._tiles.get(coord)

    def __iter__(self):
        return iter(self._tiles.items())

    def __len__(self):
        return len(self._tiles)

    @property
    def coords(self):
        return list(self._tiles.keys())

    def add(self, coord, n_values):
        self._tiles[coord] = Tile(n_values)

    def add_list(self, coord_list, n_values):
        for coord in coord_list:
            self.add(coord, n_values)

    def set_adj(self, coord_neighbors_fun):
        self._adj = {}
        self._adj_offmap = {}
        for coord, tile in self._tiles.items():
            self._adj[coord] = {}
            self._adj_offmap[coord] = {}
            for n_key, n_coord in coord_neighbors_fun(coord).items():
                if n_coord in self._tiles:
                    self._adj[coord][n_key] = n_coord
                else:
                    self._adj_offmap[coord][n_key] = n_coord

    def get_coord_adj(self, coord):
        return self._adj[coord]

    def get_coord_adj_offmap(self, coord):
        return self._adj_offmap[coord]

    def get_alpha(self, n_key, coord, n_coord):
        # alpha_{ d=n_key, i=coord, j=n_coord }: 1 if i->j are connected vid a, 0 if not
        coord_adj = self.get_coord_adj(coord)
        if n_key in coord_adj and coord_adj[n_key] == n_coord:
            return 1
        return 0

    def get_beta(self, n_key, coord):
        # beta_{ d=n_key, i=coord }: 1 if i is connected to the edge via d, 0 if not
        coord_adj_offmap = self.get_coord_adj_offmap(coord)
        if n_key in coord_adj_offmap:
            return 1
        return 0


class DirectionRule:
    def __init__(self, value_fun, weight_fun, context):
        self._value_fun = value_fun  # value: (coord) -> int [0, n_values-1]
        self._weight_fun = weight_fun  # weight: (coord, coord_adj, coord_adj_offmap, mapped_coords, context) -> weight >= 0
        self._context = context

    @property
    def value_fun(self):
        return self._value_fun

    @property
    def weight_fun(self):
        return self._weight_fun

    def evaluate_value_fun(self, coord):
        return self._value_fun(coord)

    def evaluate_weight_fun(self, coord, coord_adj, coord_adj_offmap, mapped_coords):
        return self._weight_fun(coord, coord_adj, coord_adj_offmap, mapped_coords, self._context)


class DirectionRuleSet(AlphabetUser):
    def __init__(self, n_values):
        super().__init__(n_values)
        self._rules = []  # list of DirectionRule

    def __iter__(self):
        return iter(self._rules)

    def __len__(self):
        return len(self._rules)

    def add(self, value_fun_or_const, weight_fun, context=None):
        if not callable(value_fun_or_const) and type(value_fun_or_const) is int and value_fun_or_const in range(-1,
                                                                                                                self.n_values):
            def value_fun(coord):
                return int(value_fun_or_const)
        else:
            value_fun = value_fun_or_const
        assert callable(value_fun)
        assert callable(weight_fun)
        self._rules.append(DirectionRule(value_fun, weight_fun, context))

    def provide_options(self, coord, coord_adj, coord_adj_offmap, mapped_coords):
        options = {v: 0. for v in range(self.n_values)}
        for rule in self._rules:
            w = rule.evaluate_weight_fun(coord, coord_adj, coord_adj_offmap, mapped_coords)
            v = rule.evaluate_value_fun(coord)
            options[v] += w
        options_sum = sum(options.values())
        if options_sum > 0:
            options = {v: w / options_sum for v, w in options.items()}
        else:
            options = None
        return options  # { value: probability } or None


class PCGInterface(AlphabetUser):
    def __init__(self, n_values):
        super().__init__(n_values)

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError


class WFCInterface(PCGInterface):
    def __init__(self, n_values, coord_list, coord_neighbors_fun):
        super().__init__(n_values)
        self._coord_list = coord_list
        self._coord_neighbors_fun = coord_neighbors_fun
