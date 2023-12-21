from typing import Any, Callable, Iterator, Tuple
from itertools import product
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Clbit
from qiskit.circuit.library import RYGate
from qwfc.runner import QuantumRunnerInterface
from qwfc.common import AlphabetUser, TileMap, DirectionRuleSet, WFCInterface


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

    def apply_to_circuit(self, qc, coord, optionset):
        qubits = self._encoding_qubits[coord]
        for option_idx, option in enumerate(optionset):
            coord_value_dict = option.coord_value_dict
            if option.leads_to_validity:
                all_n_qubits = self._add_x_gates(qc, coord_value_dict)
                self._add_rotation_gates(qc, qubits, all_n_qubits, option.p_dict)
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
            bitstring_parsed = {}  # mapped_coords: {coord: value}
            for coord, qubits in self._encoding_qubits.items():
                value = self._bits2value([bitstring_r[qubit] for qubit in qubits])
                bitstring_parsed[coord] = value
            if self._use_feasibility:
                feasibility = all([self._bits2value(bitstring_r[idx])==0 for idx in self._feasibility_clbits])
            else:
                feasibility = None
            parsed_result = (p, bitstring_parsed, feasibility)
            parsed_counts[bitstring] = parsed_result
        return parsed_counts


class SegmentOptions:
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


class SegmentOptionsSet(AlphabetUser):
    def __init__(self, n_values):
        super().__init__(n_values)
        self._options = []  # list of SegmentOptions

    def __iter__(self):
        return iter(self._options)

    def __len__(self):
        return len(self._options)

    def add(self, coord_value_dict, p_dict):
        epsilon = 1e-12
        assert type(coord_value_dict) is dict
        assert all([coord_value_dict != rule.coord_value_dict for rule in self._options])
        assert p_dict is None or type(p_dict) is dict
        assert p_dict is None or abs(sum(p_dict.values()) - 1) < epsilon
        assert p_dict is None or all([value in range(self._n_values) for value in p_dict.keys()])
        self._options.append(SegmentOptions(coord_value_dict, p_dict))

    def clear(self):
        self._options.clear()

    def from_direction_ruleset(self, ruleset, coord, coord_adj, coord_adj_offmap, visited_coord_adj, coord_fixed, check_feasibility):
        self.clear()
        for values in product(range(self.n_values), repeat=len(visited_coord_adj)):
            mapped_coords = {}
            for visited_index, visited_coord in enumerate(visited_coord_adj.values()):
                mapped_coords[visited_coord] = values[visited_index]
            effective_mapped_coords = {}
            effective_mapped_coords.update(coord_fixed) # choosing coord_fixed first allows that fixed coords can be overwritten
            effective_mapped_coords.update(mapped_coords)
            options = ruleset.provide_options(coord, coord_adj, coord_adj_offmap, effective_mapped_coords)
            if options is not None or check_feasibility:
                self.add(mapped_coords, options)

class QWFC(WFCInterface):
    """Create circuits for QWFC and run them to obtain tiled maps as samples."""

    def __init__(self, n_values: int, coord_list: list[tuple],
                 coord_neighbors_fun: Callable[[tuple], dict[Any, tuple]]):
        """
        :param n_values: Number of different tile IDs.
        :param coord_list: List of coordinates (tuples) the map consists of.
        :param coord_neighbors_fun: Function to obtain adjacent coordinates for every coordinate.
        """
        super().__init__(n_values, coord_list, coord_neighbors_fun)
        self._tiles = TileMapQuantumRepresentation()
        self._tiles.add_list(self._coord_list, self._n_values)
        self._tiles.set_adj(self._coord_neighbors_fun)
        self._tiles.set_qubits()
        #
        self._optionset = SegmentOptionsSet(self.n_values)
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
        return self._pc  # {bitstring: (probability, mapped_coords, feasibility)}

    def default_coord_path_fun(self, coord_list):
        return (tuple(coord) for coord in coord_list)

    def circuit(self,
                ruleset: DirectionRuleSet,
                coord_path_fun: Callable[[list[tuple]], Iterator[tuple]] = None,
                coord_fixed: dict[tuple, int] = None,
                callback_fun: Callable[[Any, int, tuple], None] = None,
                check_feasibility: bool = False,
                add_barriers: bool = False,
                add_measurement: bool = True) -> None:
        """
        Generate QQFC circuit and store it in self.qc.

        :param ruleset: DirectionRuleSet for value selection.
        :param coord_path_fun: Function to return an interator over all coordinates for segment identifier selection.
        :param coord_fixed: Coordinates with tile IDs as constraints.
        :param callback_fun: Callback function for each coordinate iteration.
        :param check_feasibility: If True, store the feasibility of configurations in an additional qubit.
        :param add_barriers: If True, add barriers to circuits between coordinate iterations.
        :param add_measurement: If True, add a final measurement of all qubits.
        :return: None
        """

        if coord_path_fun is None:
            coord_path_fun = self.default_coord_path_fun
        if coord_fixed is None:
            coord_fixed = {}
        self._pc = None
        self._qc = self._tiles.create_new_circuit(check_feasibility)
        #
        self._visited_coord_list = []
        for idx, coord in enumerate(coord_path_fun(self._coord_list)):
            assert coord in self._coord_list
            coord_adj = self._tiles.get_coord_adj(coord)
            coord_adj_offmap = self._tiles.get_coord_adj_offmap(coord)
            visited_coord_adj = {n_key: n_coord for n_key, n_coord in coord_adj.items() if n_coord in self._visited_coord_list}
            self._optionset.from_direction_ruleset(ruleset, coord, coord_adj, coord_adj_offmap, visited_coord_adj, coord_fixed, check_feasibility)
            self._tiles.apply_to_circuit(self._qc, coord, self._optionset)
            self._visited_coord_list.append(coord)
            if add_barriers:
                self._qc.barrier(range(self._qc.num_qubits))
            if callback_fun is not None:
                callback_fun(self, idx, coord)
        #
        if add_measurement:
            self._tiles.add_final_measurements(self._qc)

    def parsed_counts(self, quantum_runner: QuantumRunnerInterface) -> None:
        """
        Execute the generated QWFC circuit and store the results in self._pc.

        :param quantum_runner: Implementation of CircuitRunnerInterface with an 'execute' method to run the circuits. Returns a list of probability dictionaries.
        :return: None
        """
        assert self.qc is not None
        qc_list = [self.qc]
        probabilities_dict = quantum_runner.execute(qc_list)[0]
        self._pc = self._tiles.parse_counts(probabilities_dict)

    def run(self,
            ruleset: DirectionRuleSet,
            quantum_runner: QuantumRunnerInterface,
            coord_path_fun: Callable[[list[tuple]], Iterator[tuple]] = None,
            coord_fixed: dict[tuple, int] = None,
            callback_fun: Callable[[Any, int, tuple], None] = None) -> \
            dict[str, Tuple[float, dict[tuple, int], bool]]:
        """
        Run QWFC. Generate QWFC circuit and run it on a backend. The results represent a list of possible tile maps based on the correlation rules.

        :param ruleset: DirectionRuleSet for value selection.
        :param quantum_runner: Implementation of CircuitRunnerInterface.
        :param coord_path_fun: Function to return an interator over all coordinates for segment identifier selection.
        :param coord_fixed: Coordinates with tile IDs as constraints.
        :param callback_fun: Callback function for each coordinate iteration.
        :return: parsed counts: dictionary with keys=measured bitstrings, values=(measured probability, mapped coordinates as dict with key=tuple and value=tile ID, feasibility).
        """
        self.circuit(ruleset, coord_path_fun, coord_fixed, callback_fun, quantum_runner.check_feasibility, quantum_runner.add_barriers, quantum_runner.add_measurement)
        self.parsed_counts(quantum_runner)
        return self.pc
