from abc import abstractmethod
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options


class RunnerInterface:

    def __init__(self):
        pass

    def __repr__(self):
        return f'{self.__class__}'


class ClassicalRunnerInterface(RunnerInterface):

    def __init__(self, n_samples: int):
        super().__init__()
        self._n_samples = n_samples

    @property
    def n_samples(self):
        return self._n_samples

    @abstractmethod
    def choice(self, a: list[int], p: list[float] = None) -> int:
        raise NotImplementedError


class ClassicalRunnerDefault(ClassicalRunnerInterface):

    def __init__(self, n_samples: int, numpy_rng: np.random.Generator = None):
        super().__init__(n_samples)
        if numpy_rng is None:
            numpy_rng = np.random.RandomState()
        self._numpy_rng = numpy_rng

    @property
    def rng(self):
        return self._numpy_rng

    def choice(self, a: list[int], p: list[float] = None) -> int:
        return self._numpy_rng.choice(a, p=p, size=1)[0]


class QuantumRunnerInterface(RunnerInterface):
    """Controls the circuit execution (on simulator/real device)."""

    def __init__(self, check_feasibility: bool = False, add_barriers: bool = False, add_measurement: bool = True):
        """
        :param check_feasibility: If True, store the feasibility of configurations in an additional qubit.
        :param add_barriers: If True, add barriers to circuits (between coordinate iterations).
        :param add_measurement: If True, add a final measurement of all qubits to circuits.
        """
        super().__init__()
        self._check_feasibility = check_feasibility
        self._add_barriers = add_barriers
        self._add_measurement = add_measurement

    @property
    def check_feasibility(self):
        return self._check_feasibility

    @property
    def add_barriers(self):
        return self._add_barriers

    @property
    def add_measurement(self):
        return self._add_measurement

    @abstractmethod
    def _execute(self, qc_list: list[QuantumCircuit]) -> list[dict[str, float]]:
        """
        Run circuits on a backend using the Sampler primitive and return a list of dictionaries of probabilities.

        :param qc_list: list of QuantumCircuits to execute.
        :return: list of probabilities_dict: for each circuit a dictionary with keys=measured bitstrings, values=measured probability
        """
        raise NotImplementedError

    def execute(self, qc_list: list[QuantumCircuit]) -> list[dict[str, float]]:
        """
        Run circuits on a backend using the Sampler primitive and return a list of dictionaries of probabilities.

        :param qc_list: list of QuantumCircuits to execute.
        :return: list of probabilities_dict: for each circuit a dictionary with keys=measured bitstrings, values=measured probability
        """
        if len(qc_list) == 0:
            return []
        return self._execute([qc.copy() for qc in qc_list])


class QuantumRunnerIBMQ(QuantumRunnerInterface):
    """Controls the circuit execution for IBMQ."""

    def __init__(self, backend: Any, run_kwarg_dict: dict[str, Any] = None, check_feasibility: bool = False,
                 add_barriers: bool = False, add_measurement: bool = True):
        """
        :param backend: Qiskit Backend to run the circuits on (used in Session).
        :param run_kwarg_dict: Keyword arguments for the backend execution.
        :param check_feasibility: If True, store the feasibility of configurations in an additional qubit.
        :param add_barriers: If True, add barriers to circuits (between coordinate iterations).
        :param add_measurement: If True, add a final measurement of all qubits to circuits.
        """
        super().__init__(check_feasibility, add_barriers, add_measurement)
        self.backend = backend
        if run_kwarg_dict is None:
            run_kwarg_dict = {}
        self.run_kwarg_dict = run_kwarg_dict

    def __repr__(self):
        return f'{self.__class__}(backend={self.backend}, run_kwarg_dict={self.run_kwarg_dict})'


class QuantumRunnerIBMQRuntime(QuantumRunnerIBMQ):

    def __init__(self, backend_name: str, tp_kwarg_dict: dict[str, Any] = None, run_kwarg_dict: dict[str, Any] = None,
                 shots: int = None, runtime_service_kwarg_dict: dict[str, Any] = None,
                 options_kwarg_dict: dict[str, Any] = None, check_feasibility: bool = False, add_barriers: bool = False,
                 add_measurement: bool = True,
                 ):
        """
        :param backend_name: Qiskit Backend to run the circuits on (used in Session).
        :param tp_kwarg_dict: Transpiler keyword arguments.
        :param run_kwarg_dict: Sampler.run keyword arguments.
        :param shots: Number of shots.
        :param runtime_service_kwarg_dict: QiskitRuntimeService keyword arguments.
        :param options_kwarg_dict: Options keyword arguments.
        :param check_feasibility: If True, store the feasibility of configurations in an additional qubit.
        :param add_barriers: If True, add barriers to circuits (between coordinate iterations).
        :param add_measurement: If True, add a final measurement of all qubits to circuits.
        """
        super().__init__(backend_name, run_kwarg_dict, check_feasibility, add_barriers, add_measurement)
        if tp_kwarg_dict is None:
            tp_kwarg_dict = {}
        self.tp_kwarg_dict = tp_kwarg_dict
        if runtime_service_kwarg_dict is None:
            runtime_service_kwarg_dict = {}
        self.runtime_service_kwarg_dict = runtime_service_kwarg_dict
        if options_kwarg_dict is None:
            options_kwarg_dict = {}
        self.options_kwarg_dict = options_kwarg_dict
        self.run_kwarg_dict = run_kwarg_dict
        self.shots = shots

    def _execute(self, qc_list: list[QuantumCircuit]) -> list[dict[str, float]]:
        """
        Run circuits on a backend using the Sampler primitive and return a list of dictionaries of probabilities.

        :param qc_list: list of QuantumCircuits to execute.
        :return: list of probabilities_dict: for each circuit a dictionary with keys=measured bitstrings, values=measured probability
        """
        if len(qc_list) == 0:
            return []
        #
        service = QiskitRuntimeService(**self.runtime_service_kwarg_dict)
        options = Options(**self.options_kwarg_dict)
        service_backend = service.backend(self.backend)
        #
        if not self.tp_kwarg_dict.get('disable_transpilation', False):
            qc_list = [transpile(qc, service_backend, **self.tp_kwarg_dict) for qc in qc_list]
        #
        with Session(service=service, backend=service_backend) as session:
            sampler = Sampler(session=session, options=options, **self.run_kwarg_dict)
            results = sampler.run(qc_list, shots=self.shots).result()
        #
        probabilities_dict_list = [results.quasi_dists[qc_idx].binary_probabilities() for qc_idx in range(len(qc_list))]
        #
        return probabilities_dict_list


class QuantumRunnerIBMQAer(QuantumRunnerIBMQ):

    def __init__(self, backend: Any, run_kwarg_dict: dict[str, Any] = None, tp_kwarg_dict: dict[str, Any] = None,
                 sv_p_cutoff: float = 1e-12, check_feasibility: bool = False, add_barriers: bool = False,
                 add_measurement: bool = True):
        """
        :param backend: Qiskit Backend to run the circuits on. Available simulator backends are Aer.get_backend('qasm_simulator') (for circuits with measurements) or Aer.get_backend('statevector_simulator') (for circuits without measurements).
        :param run_kwarg_dict: Execution keyword arguments.
        :param tp_kwarg_dict: Transpiler keyword arguments.
        :param sv_p_cutoff: Cutoff probabilities below this threshold for a statevector simulation (rounding errors).
        :param check_feasibility: If True, store the feasibility of configurations in an additional qubit.
        :param add_barriers: If True, add barriers to circuits (between coordinate iterations).
        :param add_measurement: If True, add a final measurement of all qubits to circuits.
        """
        super().__init__(backend, run_kwarg_dict, check_feasibility, add_barriers, add_measurement)
        if tp_kwarg_dict is None:
            tp_kwarg_dict = {}
        self.tp_kwarg_dict = tp_kwarg_dict
        self.sv_p_cutoff = sv_p_cutoff

    def _execute(self, qc_list: list[QuantumCircuit]) -> list[dict[str, float]]:
        """
        Run circuits on a backend and return a list of dictionaries of probabilities.

        :param qc_list: list of QuantumCircuits to execute.
        :return: list of probabilities_dict: for each circuit a dictionary with keys=measured bitstrings, values=measured probability
        """
        if len(qc_list) == 0:
            return []
        #
        if hasattr(self.backend, 'name') and callable(self.backend.name):
            use_sv = self.backend.name() == 'statevector_simulator'
        else:
            use_sv = False
        if use_sv:
            for qc_idx in range(len(qc_list)):
                qc_list[qc_idx].remove_final_measurements()
        #
        qc_list = [transpile(qc, self.backend, **self.tp_kwarg_dict) for qc in qc_list]
        job = self.backend.run(qc_list, **self.run_kwarg_dict)
        #
        results = job.result()
        probabilities_dict_list = []
        for qc in qc_list:
            if use_sv:
                counts = {k: v for k, v in results.get_statevector(qc).probabilities_dict().items() if
                          v > self.sv_p_cutoff}
                norm = sum(counts.values())
                probabilities_dict = {k: v / norm for k, v in counts.items()}
            else:
                shots = sum(results.get_counts(qc).values())
                probabilities_dict = {k: v / shots for k, v in results.get_counts(qc).items()}
            probabilities_dict_list.append(probabilities_dict)
        return probabilities_dict_list


class HybridRunnerInterface(RunnerInterface):
    """Controls the execution of quantum-classical hybrid WFC."""

    def __init__(self, quantum_runner: QuantumRunnerInterface):
        """
        :param quantum_runner: Controls the circuit execution (on simulator/real device).
        """
        super().__init__()
        self._quantum_runner = quantum_runner

    @property
    def quantum_runner(self):
        return self._quantum_runner


class HybridRunnerDefault(HybridRunnerInterface):
    def __init__(self, quantum_runner: QuantumRunnerInterface):
        """
        :param quantum_runner: Controls the circuit execution (on simulator/real device).
        """
        super().__init__(quantum_runner)
