from typing import Any
from abc import abstractmethod
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options


class CircuitRunnerInterface:
    """Controls the circuit execution (on simulator/real device)."""

    def __init__(self) -> None:
        pass

    def __repr__(self):
        return f'{self.__class__}'

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
        return self._execute(qc_list)


class CircuitRunnerIBMQ(CircuitRunnerInterface):
    """Controls the circuit execution for IBMQ."""

    def __init__(self, backend: Any, run_kwarg_dict: dict[str, Any] = None) -> None:
        """
        :param backend: Qiskit Backend to run the circuits on (used in Session).
        :param run_kwarg_dict: Keyword arguments for the backend execution.
        """
        super().__init__()
        self.backend = backend
        if run_kwarg_dict is None:
            run_kwarg_dict = {}
        self.run_kwarg_dict = run_kwarg_dict

    def __repr__(self):
        return f'{self.__class__}(backend={self.backend}, run_kwarg_dict={self.run_kwarg_dict})'


class CircuitRunnerIBMQRuntime(CircuitRunnerIBMQ):

    def __init__(self, backend: Any, run_kwarg_dict: dict[str, Any] = None, runtime_service_kwarg_dict: dict[str, Any] = None,
                                  options_kwarg_dict: dict[str, Any] = None,
                                  ) -> None:
        """
        :param backend: Qiskit Backend to run the circuits on (used in Session).
        :param run_kwarg_dict: Sampler.run keyword arguments.
        :param runtime_service_kwarg_dict: QiskitRuntimeService keyword arguments.
        :param options_kwarg_dict: Options keyword arguments.
        """
        super().__init__(backend, run_kwarg_dict)
        if runtime_service_kwarg_dict is None:
            runtime_service_kwarg_dict = {}
        self.runtime_service_kwarg_dict = runtime_service_kwarg_dict
        if options_kwarg_dict is None:
            options_kwarg_dict = {}
        self.options_kwarg_dict = options_kwarg_dict
        self.run_kwarg_dict = run_kwarg_dict


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
        #
        with Session(service=service, backend=self.backend) as session:
            sampler = Sampler(session=session, options=options, **self.run_kwarg_dict)
            results = sampler.run(qc_list, shots=None).result()
        #
        probabilities_dict_list = [results.quasi_dists[qc_idx].binary_probabilities() for qc_idx in range(len(qc_list))]
        #
        return probabilities_dict_list


class CircuitRunnerIBMQAer(CircuitRunnerIBMQ):

    def __init__(self, backend: Any, run_kwarg_dict: dict[str, Any] = None, tp_kwarg_dict: dict[str, Any] = None, sv_p_cutoff: float = 1e-12) -> None:
        """
        :param backend: Qiskit Backend to run the circuits on. Available simulator backends are Aer.get_backend('qasm_simulator') (for circuits with measurements) or Aer.get_backend('statevector_simulator') (for circuits without measurements).
        :param run_kwarg_dict: Execution keyword arguments.
        :param tp_kwarg_dict: Transpiler keyword arguments.
        :param sv_p_cutoff: Cutoff probabilities below this threshold for a statevector simulation (rounding errors).
        """
        super().__init__(backend, run_kwarg_dict)
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
            qc_list = [qc.copy().remove_final_measurements() for qc in qc_list]
        #
        qc_list = [transpile(qc, self.backend, **self.tp_kwarg_dict) for qc in qc_list]
        job = self.backend.run(qc_list, **self.run_kwarg_dict)
        #
        results = job.result()
        probabilities_dict_list = []
        for qc in qc_list:
            if use_sv:
                counts = {k: v for k, v in results.get_statevector(qc).probabilities_dict().items() if v > self.sv_p_cutoff}
                norm = sum(counts.values())
                probabilities_dict = {k: v / norm for k, v in counts.items()}
            else:
                shots = sum(results.get_counts(qc).values())
                probabilities_dict = {k: v / shots for k, v in results.get_counts(qc).items()}
            probabilities_dict_list.append(probabilities_dict)
        return probabilities_dict_list