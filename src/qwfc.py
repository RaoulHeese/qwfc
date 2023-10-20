from typing import Any, Callable, Iterator, Tuple
from abc import abstractmethod
from itertools import product
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Clbit
from qiskit.circuit.library import RYGate
from runner import CircuitRunnerInterface


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

class TileMapQuantumRepresentation(TileMap):
    def __init__(self):
        super().__init__()
        #
        self._encoding_qubits = None
        self._n_encoding_qubits = None
        self._feasibility_qubits = None
        self._n_feasibility_qubits = None
        #
        self._use_feasibility = False
        self._feasibility_clbits = None

    @staticmethod
    def _value2bits(value, n_bits):
        bits = tuple(int(k) for k in bin(value)[2:])
        assert n_bits >= len(bits)
        return (0,) * (n_bits - len(bits)) + bits

    @staticmethod
    def _bits2value(bits):
        return int(''.join([str(bit) for bit in bits]), 2)

    def set_qubits(self, qubit_index=0):
        self._encoding_qubits = {}
        for coord, tile in self._tiles.items():
            self._encoding_qubits[coord] = list(range(qubit_index, qubit_index + tile.n_qubits))
            qubit_index = self._encoding_qubits[coord][-1] + 1
        self._n_encoding_qubits = qubit_index
        self._feasibility_qubits = [qubit_index]
        self._n_feasibility_qubits = 1

    def get_qubits(self, coord):
        return self._encoding_qubits[coord]

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

    def _add_x_gates(self, qc, coord_value_dict):
        all_n_qubits = []
        for n_coord, n_value in coord_value_dict.items():
            n_qubits = self._encoding_qubits[n_coord]
            bitcode = self._value2bits(n_value, len(n_qubits))
            for bit, qubit in zip(bitcode, n_qubits):
                if bit == 0:
                    qc.x(qubit)
            all_n_qubits += n_qubits
        return all_n_qubits

    def _add_feasibility_switch(self, qc, ctrl_qubits):
        # use single feasibility qubit with immediate measurements and reset
        f = self._feasibility_qubits[0]
        qc.mcx(ctrl_qubits, f)
        clbit = Clbit()
        qc.add_bits([clbit])
        clbit_index = qc.find_bit(clbit).index
        self._feasibility_clbits.append(clbit_index)
        qc.measure([f], [clbit_index])
        qc.reset(f)

    def create_new_circuit(self, use_feasibility):
        self._use_feasibility = use_feasibility
        self._feasibility_clbits = []
        n_encoding = self._n_encoding_qubits
        n_feasibility = (self._n_feasibility_qubits if self._use_feasibility else 0)
        qc = QuantumCircuit(n_encoding+n_feasibility, n_encoding)
        return qc

    def apply_to_circuit(self, qc, coord, ruleset):
        qubits = self._encoding_qubits[coord]
        for rule_idx, rule in enumerate(ruleset):
            coord_value_dict = rule.coord_value_dict
            if rule.leads_to_validity:
                all_n_qubits = self._add_x_gates(qc, coord_value_dict)
                self._add_rotation_gates(qc, qubits, all_n_qubits, rule.p_dict)
                self._add_x_gates(qc, coord_value_dict)
            else:
                if self._use_feasibility:
                    all_n_qubits = self._add_x_gates(qc, coord_value_dict)
                    self._add_feasibility_switch(qc, all_n_qubits)
                    self._add_x_gates(qc, coord_value_dict)
                else:
                    raise ValueError('ruleset requires feasibility qubit')

    def add_final_measurements(self, qc):
        if self._use_feasibility:
            cr = ClassicalRegister(bits=[qc.clbits[idx] for idx in self._feasibility_clbits], name='f')
            qc.add_register(cr)
        n_encoding = self._n_encoding_qubits
        qc.measure(range(n_encoding), range(n_encoding))

    def parse_counts(self, counts):
        parsed_counts = {}
        counts = {k: v for k, v in sorted(counts.items(), key=lambda item: -item[1])}
        for bitstring, p in counts.items():
            bitstring_r = bitstring[::-1].replace(' ', '')
            bitstring_parsed = {}  # mapped coords
            for coord, qubits in self._encoding_qubits.items():
                value = self._bits2value([bitstring_r[qubit] for qubit in qubits])
                bitstring_parsed[coord] = value
            if self._use_feasibility:
                feasibility = all([self._bits2value(bitstring_r[idx])==0 for idx in self._feasibility_clbits])
                parsed_result = (p, bitstring_parsed, feasibility)
            else:
                parsed_result = (p, bitstring_parsed)
            parsed_counts[bitstring] = parsed_result
        return parsed_counts

class CorrelationRule:
    def __init__(self, coord_value_dict, p_dict):
        self._coord_value_dict = coord_value_dict  # {coord1: required value coord1, ...}
        self._p_dict = p_dict  # {value1: probability value1, ...} or None. If None, the rule leady to an invalid state.

    @property
    def coord_value_dict(self):
        return self._coord_value_dict

    @property
    def p_dict(self):
        return self._p_dict

    @property
    def leads_to_validity(self):
        return self.p_dict is not None


class CorrelationRuleSet(AlphabetUser):
    def __init__(self, n_values):
        super().__init__(n_values)
        self._rules = [] # list of CorrelationRule

    def __iter__(self):
        return iter(self._rules)

    def __len__(self):
        return len(self._rules)

    def add(self, coord_value_dict, p_dict):
        epsilon = 1e-12
        assert p_dict is None or abs(sum(p_dict.values()) - 1) < epsilon
        assert p_dict is None or all([value in range(self._n_values) for value in p_dict.keys()])
        self._rules.append(CorrelationRule(coord_value_dict, p_dict))


class MapInterface(AlphabetUser):
    def __init__(self, n_values, coord_list, coord_neighbors_fun, check_feasibility):
        super().__init__(n_values)
        self._coord_list = coord_list
        self._coord_neighbors_fun = coord_neighbors_fun
        self._check_feasibility = check_feasibility

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError


class Map(MapInterface):
    """Create circuits for QWFC and run them to obtain tiled maps as samples."""

    def __init__(self, n_values: int, coord_list: Iterator[tuple],
                 coord_neighbors_fun: Callable[[tuple], dict[Any, tuple]],
                 check_feasibility: bool):
        """
        :param n_values: Number of different tile IDs.
        :param coord_list: List of coordinates (tuples) the map consists of.
        :param coord_neighbors_fun: Function to obtain adjacent coordinates for every coordinate.
        :param check_feasibility: If True, store the feasibility of configurations in an additional qubit.
        """
        super().__init__(n_values, coord_list, coord_neighbors_fun, check_feasibility)
        self._tiles = TileMapQuantumRepresentation()
        self._tiles.add_list(self._coord_list, self._n_values)
        self._tiles.set_adj(self._coord_neighbors_fun)
        self._tiles.set_qubits()
        #
        self._qc = None
        self._pc = None
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
    def pc(self):
        """Return quantum measurement results (after calling self.execute or self.run)."""
        return self._pc  # bitstring: (probability, mapped_coords) if check_feasibility == False or (probability, mapped_coords, feasibility) if check_feasibility == True

    def circuit(self, coord_rules_fun: Callable[
        [tuple, dict[Any, tuple], dict[Any, Any], int, dict[tuple, int]], CorrelationRuleSet],
                coord_path_fun: Callable[[Iterator[tuple]], Iterator[tuple]], coord_fixed: dict[tuple, int] = None,
                callback_fun: Callable[[Any, int, tuple], None] = None, add_barriers: bool = False,
                add_measurement: bool = True) -> None:
        """
        Generate QQFC circuit and store it in self.qc.

        :param coord_rules_fun: Function to generate a CorrelationRuleSet for each coordinate.
        :param coord_path_fun: Function to return an interator over all coordinates.
        :param coord_fixed: Coordinates with tile IDs as constraints.
        :param callback_fun: Callback function for each coordinate iteration.
        :param add_barriers: If True, add barriers to circuits between coordinate iterations.
        :param add_measurement: If True, add a final measurement of all qubits.
        :return: None
        """

        # TODO
        if coord_fixed is None:
            coord_fixed = {}
        self._pc = None
        self._qc = self._tiles.create_new_circuit(self._check_feasibility)
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
                self._qc.barrier(range(self._qc.num_qubits))
            if callback_fun is not None:
                callback_fun(self, idx, coord)
        #
        if add_measurement:
            self._tiles.add_final_measurements(self._qc)

    def parsed_counts(self, circuit_runner: CircuitRunnerInterface) -> None:
        """
        Execute the generated QWFC circuit and store the results in self._pc.

        :param circuit_runner: Implementation of CircuitRunnerInterface with an 'execute' method to run the circuits. Returns a list of probability dictionaries.
        :return: None
        """
        assert self.qc is not None
        qc_list = [self.qc]
        probabilities_dict = circuit_runner.execute(qc_list)[0]
        self._pc = self._tiles.parse_counts(probabilities_dict)

    def run(self, coord_rules_fun: Callable[
        [tuple, dict[Any, tuple], dict[Any, Any], int, dict[tuple, int]], CorrelationRuleSet],
            coord_path_fun: Callable[[Iterator[tuple]], Iterator[tuple]],
            circuit_runner: CircuitRunnerInterface,
            coord_fixed: dict[tuple, int] = None,
            callback_fun: Callable[[Any, int, tuple], None] = None, add_barriers: bool = False, add_measurement: bool = True) -> \
            dict[str, Tuple[float, dict[tuple, int]]]:
        """
        Generate QWFC circuit and run it on a backend. The results represent a list of possible tile maps based on the correlation rules.

        :param coord_rules_fun: Function to generate a CorrelationRuleSet for each coordinate.
        :param coord_path_fun: Function to return an interator over all coordinates.
        :param circuit_runner: Implementation of CircuitRunnerInterface with an 'execute' method to run the circuits. Returns a list of probability dictionaries.
        :param coord_fixed: Coordinates with tile IDs as constraints.
        :param callback_fun: Callback function for each coordinate iteration.
        :param add_barriers: If True, add barriers to circuits between coordinate iterations for the created circuit.
        :param add_measurement: If True, add a final measurement of all qubits for the created circuit.
        :return: parsed counts: dictionary with keys=measured bitstrings, values=(measured probability, mapped coordinates as dict with key=tuple and value=tile ID).
        """
        self.circuit(coord_rules_fun, coord_path_fun, coord_fixed, callback_fun, add_barriers, add_measurement)
        self.parsed_counts(circuit_runner)
        return self.pc


class MapSlidingWindow(MapInterface):
    """Sliding window approach to split a big tile map into smaller maps that can be handled with QWFC."""

    def __init__(self, n_values: int, coord_list: Iterator[tuple],
                 coord_neighbors_fun: Callable[[tuple], dict[Any, tuple]], check_feasibility: bool):
        """
        :param n_values: Number of different tile IDs.
        :param coord_list: List of coordinates (tuples) the map consists of.
        :param coord_neighbors_fun: Function to obtain adjacent coordinates for every coordinate.
        :param check_feasibility: If True, store the feasibility of configurations in an additional qubit.
        """
        super().__init__(n_values, coord_list, coord_neighbors_fun, check_feasibility)
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
            circuit_runner: CircuitRunnerInterface,
            segment_callback_fun: Callable[[Any, int, tuple], None] = None,
            callback_fun: Callable[[Any, int, Map, dict[tuple, int]], None] = None, add_barriers: bool = False, add_measurement: bool = True) -> None:
        """
        Use a sliding window approach to generate multiple QWFC circuits and run each of them on a backend. Compose these results to obtain a tile map.

        :param segment_map_fun: Function to extract a single tile map from the resulting list of tile maps from a QWFC circuit execution.
        :param segment_iter_fun: Function to obtain list of coordinates and (function to generate) iterators over these coordinates for each segment.
        :param coord_rules_fun: Function to generate a CorrelationRuleSet for each coordinate.
        :param circuit_runner: Implementation of CircuitRunnerInterface with an 'execute' method to run the circuits.
        :param segment_callback_fun: Callback function for each coordinate iteration within the segment evaluation.
        :param callback_fun: Callback function for each segment iteration.
        :param add_barriers: If True, add barriers to circuits between coordinate iterations for the created circuit.
        :param add_measurement: If True, add a final measurement of all qubits for the created circuit.
        :return: None
        """
        self._mapped_coords = {}
        #
        for idx, (coord_list, coord_path_fun) in enumerate(segment_iter_fun(self._coord_list)):
            map_segment = Map(self._n_values, coord_list, self._coord_neighbors_fun, self._check_feasibility)
            map_segment.run(coord_rules_fun, coord_path_fun,
                            circuit_runner=circuit_runner,
                            coord_fixed = self._mapped_coords,
                            callback_fun=segment_callback_fun,
                            add_barriers=add_barriers, add_measurement=add_measurement)
            segment_mapped_coords = segment_map_fun(map_segment.pc)
            self._mapped_coords.update(segment_mapped_coords)
            if callback_fun is not None:
                callback_fun(self, idx, map_segment, segment_mapped_coords)
