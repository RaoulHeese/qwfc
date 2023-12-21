from typing import Any, Callable, Iterator, Tuple
from qwfc.common import DirectionRuleSet, TileMap, WFCInterface
from qwfc.runner import ClassicalRunnerInterface
import numpy as np

class TileMapClassicalRepresentation(TileMap):
    def __init__(self):
        super().__init__()

class CWFC(WFCInterface):
    def __init__(self, n_values: int, coord_list: list[tuple],
                 coord_neighbors_fun: Callable[[tuple], dict[Any, tuple]]):
        """
        :param n_values: Number of different tile IDs.
        :param coord_list: List of coordinates (tuples) the map consists of.
        :param coord_neighbors_fun: Function to obtain adjacent coordinates for every coordinate.
        """
        super().__init__(n_values, coord_list, coord_neighbors_fun)
        self._tiles = TileMapClassicalRepresentation()
        self._tiles.add_list(self._coord_list, self._n_values)
        self._tiles.set_adj(self._coord_neighbors_fun)
        #
        self._pc = None
        self._mapped_coords = None  # content: {coord: value}
        self._feasibility = None

    @property
    def pc(self):
        """Return wfc results."""
        return self._pc  # {key: (probability, mapped_coords, feasibility)}

    @property
    def mapped_coords(self):
        """Return mapped coordinates."""
        return self._mapped_coords  # {coord: value}

    def coord_path_fun(self, ruleset, classical_runner):
        # minimum entropy
        coord_candidates = {}
        for coord in self._tiles.coords:
            if coord not in self._mapped_coords:
                coord_adj = self._tiles.get_coord_adj(coord)
                coord_adj_offmap = self._tiles.get_coord_adj_offmap(coord)
                options = ruleset.provide_options(coord, coord_adj, coord_adj_offmap, self._mapped_coords)
                if options is None:
                    return None
                p = np.array(list(options.values()))
                p[p==0] = 1
                entropy = - np.dot(p, np.log(p))
                coord_candidates[coord] = entropy
        if len(coord_candidates) > 0:
            lowest_entropy = min(list(coord_candidates.values()))
            coord_candidate_list = [coord for coord, entropy in coord_candidates.items() if entropy == lowest_entropy]
            coord_idx = classical_runner.choice(list(range(len(coord_candidate_list))))
            coord = coord_candidate_list[coord_idx]
        else:
            coord = None
        return coord

    def sample(self, ruleset, classical_runner, coord_fixed=None, callback_fun=None):
        if coord_fixed is None:
            coord_fixed = {}
        self._mapped_coords = {}
        self._mapped_coords.update(coord_fixed)
        self._feasibility = True
        #
        while len(self._tiles) > len(self._mapped_coords):
            coord = self.coord_path_fun(ruleset, classical_runner)
            if coord is None:
                self._feasibility = False
                break
            #
            coord_adj = self._tiles.get_coord_adj(coord)
            coord_adj_offmap = self._tiles.get_coord_adj_offmap(coord)
            effective_mapped_coords = {}
            effective_mapped_coords.update(self._mapped_coords)
            effective_mapped_coords.update(coord_fixed)
            options = ruleset.provide_options(coord, coord_adj, coord_adj_offmap, effective_mapped_coords)
            if options is None:
                self._feasibility = False
                break
            #
            new_value = classical_runner.choice(list(options.keys()), p=list(options.values()))
            self._mapped_coords[coord] = new_value
            #
            if callback_fun is not None:
                idx = len(self._mapped_coords)
                callback_fun(self, idx, coord, options, new_value)

    def run(self,
            ruleset: DirectionRuleSet,
            classical_runner: ClassicalRunnerInterface,
            coord_fixed: dict[tuple, int] = None,
            callback_fun: Callable[[Any, int, tuple, dict[int, float], int], None] = None) -> \
            dict[Any, Tuple[float, dict[tuple, int], bool]]:

        """
        Run CWFC.

        :param ruleset: DirectionRuleSet for value selection.
        :param classical_runner: Implementation of ClassicalRunnerInterface.
        :param coord_fixed: Coordinates with tile IDs as constraints.
        :param callback_fun: Callback function for each coordinate iteration.
        :return:
        """

        samples = {}
        for n in range(classical_runner.n_samples):
            self.sample(ruleset, classical_runner, coord_fixed, callback_fun)
            key = tuple([self._mapped_coords.get(coord, None) for coord in self._tiles.coords])
            if key not in samples:
                samples[key] = {'counts': 1, 'mapped_coords': self._mapped_coords, 'feasibility': self._feasibility}
            else:
                samples[key]['counts'] += 1
                assert self._feasibility == samples[key]['feasibility']

        self._pc = {}  # { key : (p, mc, f) }
        norm = sum([samples[key]['counts'] for key in samples.keys()])
        for key, sample in samples.items():
            self._pc[key] = (sample['counts']/norm, sample['mapped_coords'], sample['feasibility'])
        return self.pc
