from typing import Any, Callable, Iterator, Tuple
from abc import abstractmethod
from itertools import product
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate


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
        self._adj = None
        self._adj_offmap = None
        #
        self._qubits = None
        self._n_qubits = None

    @property
    def n_qubits(self):
        return self._n_qubits

    @staticmethod
    def _value2bits(value, n_bits):
        bits = tuple(int(k) for k in bin(value)[2:])
        assert n_bits >= len(bits)
        return (0,) * (n_bits - len(bits)) + bits

    @staticmethod
    def _bits2value(bits):
        return int(''.join([str(bit) for bit in bits]), 2)

    def __getitem__(self, coord):
        return self._tiles.get(coord)

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

    def get_adj(self, coord):
        return self._adj[coord]

    def get_adj_offmap(self, coord):
        return self._adj_offmap[coord]

    def set_qubits(self, qubit_index=0):
        self._qubits = {}
        for coord, tile in self._tiles.items():
            self._qubits[coord] = list(range(qubit_index, qubit_index + tile.n_qubits))
            qubit_index = self._qubits[coord][-1] + 1
        self._n_qubits = qubit_index

    def get_qubits(self, coord):
        return self._qubits[coord]

    def _add_rotation_gates(self, qc, target_qubits, ctrl_qubits, p_dict):
        def p_sub(p_bit_dict_, conditions):
            return np.sum(
                [value for key, value in p_bit_dict_.items() if all([key[k_] == v for k_, v in conditions.items()])])

        p_bit_dict = {self._value2bits(value, len(target_qubits)): p for value, p in
                      p_dict.items()}  # p(qubit_i = bitcode_i)
        for n in range(len(target_qubits)):
            for k, startbits in enumerate(product((0, 1), repeat=n)):
                p_k_top = p_sub(p_bit_dict, {idx: bit for idx, bit in enumerate(startbits + (0,))})
                p_k_bottom = p_sub(p_bit_dict, {idx: bit for idx, bit in enumerate(startbits)}) if len(
                    startbits) > 0 else 1
                if p_k_top == p_k_bottom == 0:
                    p_k = 0
                else:
                    p_k = p_k_top / p_k_bottom
                theta = 2 * np.arccos(np.sqrt(p_k))
                for idx, bit in enumerate(startbits):
                    if bit == 0:
                        qc.x(target_qubits[idx])
                effective_ctrl_qubits = ctrl_qubits + [target_qubits[idx] for idx in range(n)]
                effective_target_qubits = [target_qubits[n]]
                if len(effective_ctrl_qubits) > 0:
                    qc.append(RYGate(theta).control(len(effective_ctrl_qubits)),
                              effective_ctrl_qubits + effective_target_qubits)
                else:
                    qc.append(RYGate(theta), effective_target_qubits)
                for idx, bit in enumerate(startbits):
                    if bit == 0:
                        qc.x(target_qubits[idx])

    def apply_to_circuit(self, qc, coord, ruleset):
        qubits = self._qubits[coord]
        for rule_idx, rule in enumerate(ruleset):
            coord_value_dict = rule.coord_value_dict
            p_dict = rule.p_dict
            all_n_qubits = []
            for n_coord, n_value in coord_value_dict.items():
                n_qubits = self._qubits[n_coord]
                bitcode = self._value2bits(n_value, len(n_qubits))
                for bit, qubit in zip(bitcode, n_qubits):
                    if bit == 0:
                        qc.x(qubit)
                all_n_qubits += n_qubits
            self._add_rotation_gates(qc, qubits, all_n_qubits, p_dict)
            for n_coord, n_value in coord_value_dict.items():
                n_qubits = self._qubits[n_coord]
                bitcode = self._value2bits(n_value, len(n_qubits))
                for bit, qubit in zip(bitcode, n_qubits):
                    if bit == 0:
                        qc.x(qubit)

    def parse_counts(self, counts):
        parsed_counts = {}
        counts = {k: v for k, v in sorted(counts.items(), key=lambda item: -item[1])}
        for bitstring, p in counts.items():
            bitstring_r = bitstring[::-1]
            bitstring_parsed = {}  # mapped coords
            for coord, qubits in self._qubits.items():
                value = self._bits2value([bitstring_r[qubit] for qubit in qubits])
                bitstring_parsed[coord] = value
            parsed_counts[bitstring] = (p, bitstring_parsed)
        return parsed_counts


class CorrelationRule:
    def __init__(self, coord_value_dict, p_dict):
        self._coord_value_dict = coord_value_dict  # {coord1: required value coord1, ...}
        self._p_dict = p_dict  # {value1: probability value1, ...}

    @property
    def coord_value_dict(self):
        return self._coord_value_dict

    @property
    def p_dict(self):
        return self._p_dict


class CorrelationRuleSet(AlphabetUser):
    def __init__(self, n_values):
        super().__init__(n_values)
        self._rules = []

    def __iter__(self):
        return iter(self._rules)

    def __len__(self):
        return len(self._rules)

    def add(self, coord_value_dict, p_dict):
        epsilon = 1e-12
        assert abs(sum(p_dict.values()) - 1) < epsilon
        assert all([value in range(self._n_values) for value in p_dict.keys()])
        self._rules.append(CorrelationRule(coord_value_dict, p_dict))


class MapInterface(AlphabetUser):
    def __init__(self, n_values, coord_list, coord_neighbors_fun):
        super().__init__(n_values)
        self._coord_list = coord_list
        self._coord_neighbors_fun = coord_neighbors_fun

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError


class Map(MapInterface):
    """Create circuits for QWFC and run them to obtain tiled maps as samples."""

    def __init__(self, n_values: int, coord_list: Iterator[tuple],
                 coord_neighbors_fun: Callable[[tuple], dict[Any, tuple]]):
        """
        :param n_values: Number of different tile IDs.
        :param coord_list: List of coordinates (tuples) the map consists of.
        :param coord_neighbors_fun: Function to obtain adjacent coordinates for every coordinate.
        """
        super().__init__(n_values, coord_list, coord_neighbors_fun)
        self._tiles = TileMap()
        self._tiles.add_list(self._coord_list, self._n_values)
        self._tiles.set_adj(self._coord_neighbors_fun)
        self._tiles.set_qubits()
        #
        self._qc = None
        self._parsed_counts = None
        self._visited_coord_list = None

    @property
    def tiles(self):
        """Return tile map (TileMap instance)."""
        return self._tiles

    @property
    def qc(self):
        """Return generated quantum circuit (after calling self.circuit or self.run)."""
        return self._qc

    @property
    def parsed_counts(self):
        """Return quantum measurement results (after calling self.execute or self.run)."""
        return self._parsed_counts  # bitstring: (probability, mapped_coords)

    def circuit(self, coord_rules_fun: Callable[
        [tuple, dict[Any, tuple], dict[Any, Any], int, dict[tuple, int]], CorrelationRuleSet],
                coord_path_fun: Callable[[Iterator[tuple]], Iterator[tuple]], coord_fixed: dict[tuple, int] = None,
                callback_fun: Callable[[Any, int, tuple], None] = None, add_barriers: bool = False,
                use_sv: bool = False):
        """
        Generate QQFC circuit and store it in self.qc.

        :param coord_rules_fun: Function to generate a CorrelationRuleSet for each coordinate.
        :param coord_path_fun: Function to return an interator over all coordinates.
        :param coord_fixed: Coordinates with tile IDs as constraints.
        :param callback_fun: Callback function for each coordinate iteration.
        :param add_barriers: If True, add barriers to circuits between coordinate iterations.
        :param use_sv: If True, presume statevector simulation.
        :return: None
        """
        if coord_fixed is None:
            coord_fixed = {}
        n = self._tiles.n_qubits
        self._parsed_counts = None
        self._qc = QuantumCircuit(n, n)
        #
        self._visited_coord_list = []
        for idx, coord in enumerate(coord_path_fun(self._coord_list)):
            assert coord in self._coord_list
            coord_adj = self._tiles.get_adj(coord)
            coord_adj_offmap = self._tiles.get_adj_offmap(coord)
            visited_coord_adj = {n_key: n_coord for n_key, n_coord in coord_adj.items() if
                                 n_coord in self._visited_coord_list}
            ruleset = coord_rules_fun(coord, visited_coord_adj, coord_adj_offmap, self._n_values, coord_fixed)
            self._tiles.apply_to_circuit(self._qc, coord, ruleset)
            self._visited_coord_list.append(coord)
            if add_barriers:
                self._qc.barrier(range(n))
            if callback_fun is not None:
                callback_fun(self, idx, coord)
        #
        if use_sv:
            self._qc.save_statevector()
        self._qc.measure(range(n), range(n))

    def execute(self, backend: Any, tp_kwarg_dict: dict[str, Any] = None, run_kwarg_dict: dict[str, Any] = None,
                use_sv: bool = False, sv_p_cutoff: float = 1e-12):
        """
        Run generated QQFC circuit on a backend and store the results in self.parsed_counts.

        :param backend: Qiskit Backend to run the circuit on.
        :param tp_kwarg_dict: Transpiler keyword arguments.
        :param run_kwarg_dict: Execution keyword arguments.
        :param use_sv: If True, presume statevector simulation.
        :param sv_p_cutoff: Cutoff probabilities below this threshold for a statevector simulation (rounding errors).
        :return: None
        """

        assert self.qc is not None
        if tp_kwarg_dict is None:
            tp_kwarg_dict = {}
        if run_kwarg_dict is None:
            run_kwarg_dict = {}
        qc = transpile(self.qc, backend, **tp_kwarg_dict)
        job = backend.run(qc, **run_kwarg_dict)
        result = job.result()
        if use_sv:
            counts = {k: v for k, v in result.get_statevector(qc).probabilities_dict().items() if v > sv_p_cutoff}
            norm = sum(counts.values())
            counts = {k: v / norm for k, v in counts.items()}
        else:
            shots = sum(result.get_counts(qc).values())
            counts = {k: v / shots for k, v in result.get_counts(qc).items()}
        self._parsed_counts = self._tiles.parse_counts(counts)

    def run(self, coord_rules_fun: Callable[
        [tuple, dict[Any, tuple], dict[Any, Any], int, dict[tuple, int]], CorrelationRuleSet],
            coord_path_fun: Callable[[Iterator[tuple]], Iterator[tuple]],
            backend: Any, coord_fixed: dict[tuple, int] = None,
            callback_fun: Callable[[Any, int, tuple], None] = None, add_barriers: bool = False, use_sv: bool = False,
            tp_kwarg_dict: dict[str, Any] = None, run_kwarg_dict: dict[str, Any] = None, sv_p_cutoff: float = 1e-12) -> \
            dict[str, Tuple[float, dict[tuple, int]]]:
        """
        Generate QWFC circuit and run it on a backend. The results represent a list of possible tile maps based on the correlation rules.

        :param coord_rules_fun: Function to generate a CorrelationRuleSet for each coordinate.
        :param coord_path_fun: Function to return an interator over all coordinates.
        :param backend: Qiskit Backend to run the circuit on.
        :param coord_fixed: Coordinates with tile IDs as constraints.
        :param callback_fun: Callback function for each coordinate iteration.
        :param add_barriers: If True, add barriers to circuits between coordinate iterations.
        :param use_sv: If True, presume statevector simulation.
        :param tp_kwarg_dict: Transpiler keyword arguments.
        :param run_kwarg_dict: Execution keyword arguments.
        :param sv_p_cutoff: Cutoff probabilities below this threshold for a statevector simulation (rounding errors).
        :return: parsed counts: dictionary with keys=measured bitstrings, values=(measured probability, mapped coordinates as dict with key=tuple and value=tile ID).
        """
        self.circuit(coord_rules_fun, coord_path_fun, coord_fixed, callback_fun, add_barriers, use_sv)
        self.execute(backend, tp_kwarg_dict, run_kwarg_dict, use_sv, sv_p_cutoff)
        return self.parsed_counts


class MapSlidingWindow(MapInterface):
    """Sliding window approach to split a big tile map into smaller maps that can be handled with QWFC."""

    def __init__(self, n_values: int, coord_list: Iterator[tuple],
                 coord_neighbors_fun: Callable[[tuple], dict[Any, tuple]]):
        """
        :param n_values: Number of different tile IDs.
        :param coord_list: List of coordinates (tuples) the map consists of.
        :param coord_neighbors_fun: Function to obtain adjacent coordinates for every coordinate.
        """
        super().__init__(n_values, coord_list, coord_neighbors_fun)
        self._mapped_coords = None

    @property
    def mapped_coords(self):
        """Return mapped coordinates as dict with key=tuple and value=tile ID (after calling self.run)."""
        return self._mapped_coords

    def run(self, segment_map_fun: Callable[[dict[str, Tuple[float, dict[tuple, int]]]], dict[tuple, int]],
            segment_iter_fun: Callable[
                [Iterator[tuple]], Tuple[Iterator[tuple], Callable[[Iterator[tuple]], Iterator[tuple]]]],
            coord_rules_fun: Callable[
                [tuple, dict[Any, tuple], dict[Any, Any], int, dict[tuple, int]], CorrelationRuleSet],
            backend: Any, tp_kwarg_dict: dict[str, Any] = None, run_kwarg_dict: dict[str, Any] = None,
            use_sv: bool = False, segment_callback_fun: Callable[[Any, int, tuple], None] = None,
            callback_fun: Callable[[Any, int, Map, dict[tuple, int]], None] = None):
        """
        Use a sliding window approach to generate multiple QWFC circuits and run each of them on a backend. Compose these results to obtain a tile map.

        :param segment_map_fun: Function to extract a single tile map from the resulting list of tile maps from a QWFC circuit execution.
        :param segment_iter_fun: Function to obtain list of coordinates and (function to generate) iterators over these coordinates for each segment.
        :param coord_rules_fun: Function to generate a CorrelationRuleSet for each coordinate.
        :param backend: Qiskit Backend to run the circuit on.
        :param tp_kwarg_dict: Transpiler keyword arguments.
        :param run_kwarg_dict: Execution keyword arguments.
        :param use_sv: If True, presume statevector simulation.
        :param segment_callback_fun: Callback function for each coordinate iteration within the segment evaluation.
        :param callback_fun: Callback function for each segment iteration.
        :return:
        """
        self._mapped_coords = {}
        #
        for idx, (coord_list, coord_path_fun) in enumerate(segment_iter_fun(self._coord_list)):
            map_segment = Map(self._n_values, coord_list, self._coord_neighbors_fun)
            map_segment.run(coord_rules_fun, coord_path_fun, backend, coord_fixed=self._mapped_coords,
                            callback_fun=segment_callback_fun,
                            use_sv=use_sv, tp_kwarg_dict=tp_kwarg_dict, run_kwarg_dict=run_kwarg_dict)
            segment_mapped_coords = segment_map_fun(map_segment.parsed_counts)
            self._mapped_coords.update(segment_mapped_coords)
            if callback_fun is not None:
                callback_fun(self, idx, map_segment, segment_mapped_coords)
