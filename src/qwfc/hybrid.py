from typing import Any, Callable, Tuple, Iterator

from qwfc.common import WFCInterface, DirectionRuleSet
from qwfc.quantum import QWFC
from qwfc.runner import HybridRunnerInterface


class HWFC(WFCInterface):
    """Sliding window approach to split a big tile map into smaller maps that can be handled with QWFC."""

    def __init__(self, n_values: int, coord_list: list[tuple],
                 coord_neighbors_fun: Callable[[tuple], dict[Any, tuple]]):
        """
        :param n_values: Number of different tile IDs.
        :param coord_list: List of coordinates (tuples) the map consists of.
        :param coord_neighbors_fun: Function to obtain adjacent coordinates for every coordinate.
        """
        super().__init__(n_values, coord_list, coord_neighbors_fun)
        #
        self._pc = None
        self._mapped_coords = None
        self._feasibility = None

    @property
    def pc(self):
        """Return wfc results."""
        return self._pc  # {key: (probability, mapped_coords, feasibility)}

    @property
    def mapped_coords(self):
        """Return mapped coordinates as dict with key=tuple and value=tile ID (after calling self.run)."""
        return self._mapped_coords

    def run(self,
            ruleset: DirectionRuleSet,
            hybrid_runner: HybridRunnerInterface,
            chunk_map_fun: Callable[[dict[str, Tuple[float, dict[tuple, int]]]], dict[tuple, int]],
            chunk_iter_fun: Callable[
                [Iterator[tuple]], Tuple[Iterator[tuple], Callable[[Iterator[tuple]], Iterator[tuple]]]],
            qwfc_callback_fun: Callable[[Any, int, tuple], None] = None,
            hwfc_callback_fun: Callable[[Any, int, QWFC, dict[tuple, int]], None] = None) -> None:
        """
        Run HWFC. Use a sliding window approach to generate multiple QWFC circuits and run each of them on a backend. Compose these results to obtain a tile map.

        :param ruleset: DirectionRuleSet for value selection.
        :param hybrid_runner: Implementation of HybridRunnerInterface.
        :param chunk_map_fun: Function to extract a single tile map from the resulting list of tile maps from a QWFC circuit execution.
        :param chunk_iter_fun: Function to obtain list of coordinates and (function to generate) iterators over these coordinates for each chunk.
        :param qwfc_callback_fun: Callback function for each coordinate iteration within the chunk evaluation.
        :param hwfc_callback_fun: Callback function for each chunk iteration.
        :return: None
        """
        self._mapped_coords = {}
        self._feasibility = True if hybrid_runner.quantum_runner.check_feasibility else None
        #
        for idx, (coord_list, coord_path_fun) in enumerate(chunk_iter_fun(self._coord_list)):
            map_chunk = QWFC(self._n_values, coord_list, self._coord_neighbors_fun)
            map_chunk.run(ruleset, quantum_runner=hybrid_runner.quantum_runner,
                          coord_path_fun=coord_path_fun,
                          coord_fixed=self._mapped_coords,
                          callback_fun=qwfc_callback_fun)
            chunk_mapped_coords = chunk_map_fun(map_chunk.pc)
            if chunk_mapped_coords is None:
                self._feasibility = False
                break
            else:
                self._mapped_coords.update(chunk_mapped_coords)
            if hwfc_callback_fun is not None:
                hwfc_callback_fun(self, idx, map_chunk, chunk_mapped_coords)
        #
        self._pc = {'hybrid': (1, self._mapped_coords, self._feasibility)}
        return self.pc
