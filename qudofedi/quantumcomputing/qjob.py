from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import CircuitSampler, CircuitStateFn, StateFn, PauliExpectation, PauliSumOp
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Estimator
from qiskit.utils import QuantumInstance
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_provider.ibm_backend import IBMBackend
from qiskit_aer.backends.qasm_simulator import QasmSimulator
import numpy as np
import pickle
import json
import io
import warnings
import os
import string
import random
import shutil
from datetime import datetime
from typing import Any
from qudofedi.experiment_info import FeynmanDiagram, System
from .circuits import LinearSpectroscopyCircuit, ThirdOrderSpectroscopyCircuit

def load(name: str,
         ):
    '''Load the information about a Qjob from a .json file.
    
    Input:
    - name: str
        Name of the origin file.
    '''
    job = Qjob().load(name)

    return job

class Qjob():
    _linear_name_list = ["a", "absorption"]
    _thirdorder_name_list = ["gsb", "ground state bleaching", "se", "stimulated emission", "esa", "excited state absorption"]

    def __reset_computational_details(self,
                                      ):
        computational_details = {"Backend": None,
                                 "Shots": None,
                                 "Noise Model": None,
                                 "Coupling Map": None,
                                 "Basis Gates": None,
                                 "Tags": None,
                                 "Starting Time": None,
                                 "Ending Time": None,
                                 "Computation Time": None,
                                 }
        return computational_details

    def __init__(self,
                 system: System | None = None,
                 feynman_diagram: FeynmanDiagram | None = None,
                 ):
        try:
            self.FD_type = feynman_diagram.type
            self.delay_time = feynman_diagram.delay_time
        except:
            self.FD_type = None
            self.delay_time = None
        self.system = system
        self.computational_details = self.__reset_computational_details()
        self.response_function = None

    def get_circuits(self,
                     ) -> list[QuantumCircuit]:
        '''Return a list of qiskit.QuantumCircuit associated to the Feynman Diagram of the Qjob object.
        
        Input:
        - system: System
            The system of interest for the Feynamn Diagram.
        - index: int
            Index of the Feynman Diagram to be returned.

        Return:
        list[QuantumCircuit]
        '''
        # Retriving the circuits.
        if self.FD_type in self._linear_name_list:
            qcs = LinearSpectroscopyCircuit(self.system)[0]
        else: 
            qcs = ThirdOrderSpectroscopyCircuit(self.FD_type, self.system)[0]
        
        return qcs

    def show_circuit(self,
                     index: list[int] | int = None,
                     ):
        '''Print the quantum circuit(s) associated with the Feynman Diagram of the Qjob object.

        Input:
        - system: System
            The system of interest for the Feynamn Diagram.
        - index: Union[list[int], int]
            Index of the Feynman Diagram to be printed.
        '''
        # Retriving the circuits.
        qcs = self.get_circuits(self.system)
        
        # Checking what to print
        if self.system.system_size == 1:
            if index != 0 and index != None:
                warnings.warn("Index is not considered as there is only 1 parametrized circuit.")
            index = 0
        else:
            if index == None:
                for qc in qcs:
                    print(qc.draw())
                return
            if isinstance(index, list):
                for i, qc in enumerate(qcs):
                    if i in index:
                        print(qc.draw())
                return
            if index >= len(qcs):
                raise IndexError('Index exceeds the number of circuits. Must be lower than ' + str(len(qcs)))

        return qcs[index].draw(output='mpl')

    def to_dict(self,
                ) -> dict:
        '''Return a dictionary containing the information about the Qjob.'''
        qjob_dict = {"System Size": self.system.system_size,
                     "System Hamiltonian": self.system.Hamiltonian,
                     "System Dipole Moment Amplitudes": self.system.dipole_moment,
                     "Feynman Diagram Type": self.FD_type,
                     "Delay Time": self.delay_time,
                     "Computational details": self.computational_details,
                     "Response Function": self.response_function,
                     }
        
        return qjob_dict
    
    def from_dict(self,
                  qjob_dict: dict,
                  ):
        '''Create the Qjob object from a dictionary.

        Input:
        - qjob_dict: dict
            A dictionary used as the base for the Qjob.
        '''
        self.FD_type = qjob_dict.pop("Feynman Diagram Type", None)
        self.delay_time = qjob_dict.pop("Delay Time", None)
        self.system = System(qjob_dict.pop("System Hamiltonian", None),
                            qjob_dict.pop("System Dipole Moment Amplitudes", None),
                            )
        self.computational_details = qjob_dict.pop("Computational details", self.__reset_computational_details())
        self.response_function = qjob_dict.pop("Response Function", None)

        return self

    def save(self,
             name: str,
             ):
        '''Save the information about the Qjob.
        
        Input:
        - name: str
            Name of the destination directory.
        '''
        try:
            # Creating the directory.
            os.mkdir(name)
        except:
            pass

        path = os.path.join(os.getcwd(), name)
        data = self.to_dict()

        # Saving the computational details
        with open(os.path.join(path, "compdet_" + name + ".pkl"),
                  "wb",
                  ) as file:
            compdet = data["Computational details"]
            pickle.dump(compdet,
                        file,
                        protocol = -1)

        # Saving the rest of data
        data.pop("Computational details")
        with open(os.path.join(path, "data_" + name + ".pkl"),
                  "wb",
                  ) as file:
            pickle.dump(data,
                        file,
                        protocol=-1)
        
        print("File saved in " + path)

    def load(self,
             name: str,
             exclude_compdet = True,
             ):
        '''Load the information about a Qjob from a directory with files.
        
        Input:
        - name: str
            Name of the origin directory.
        '''
        try:
            path = os.path.join(os.getcwd(), name)
        except Exception as err:
            raise Exception("Error during loading: ", err) 
        
        qjob_dict = {}

        # Loading the computational details
        if not exclude_compdet:
            try:
                temp_dict = {}
                with open(os.path.join(path, "compdet_" + name + ".pkl"),
                          "rb",
                          ) as file:
                    temp_dict = pickle.load(file)
                qjob_dict.update(temp_dict)
            except Exception as err:
                warnings.WarningMessage("Error " + err + " when trying to retrieve computational details from directory " + path)
        
        # Loading the rest of data
        try:
            temp_dict = {}
            with open(os.path.join(path, "data_" + name + ".pkl"),
                      "rb",
                      ) as file:
                temp_dict = pickle.load(file)
            qjob_dict.update(temp_dict)
        except Exception as err:
            warnings.WarningMessage("Error " + err + " when trying to retrieve data from directory " + path)

        self.from_dict(qjob_dict)

        return self

    def print(self,
              ) -> None:
        print(self.to_dict())

    def __QuantumInstance_generator(self,
                                    **kwargs,
                                    ) -> QuantumInstance:
        try:
            backend_name = self.computational_details["Backend"].name()
        except:
            backend_name = self.computational_details["Backend"].name
        if backend_name != "qasm_simulator":
            q_instance = QuantumInstance(backend = self.computational_details["Backend"],
                                         shots = self.computational_details["Shots"],
                                         **kwargs,
                                         )
        else:
            q_instance = QuantumInstance(backend = self.computational_details["Backend"],
                                         shots = self.computational_details["Shots"],
                                         noise_model = self.computational_details["Noise Model"],
                                         coupling_map = self.computational_details["Coupling Map"],
                                         basis_gates = self.computational_details["Basis Gates"],
                                         **kwargs,
                                         )
        return q_instance
    
    def __Options_generator(self,
                            **kwargs,
                            ):
        if self.computational_details["Noise Model"] == None:
            options = Options(**kwargs)
        else:
            options = Options(simulator={"noise_model": self.computational_details["Noise Model"],
                                         "coupling_map": self.computational_details["Coupling Map"],
                                         "basis_gates": self.computational_details["Basis Gates"]},
                                         **kwargs,
                              )
        return options

    def __LinearSpectroscopy(self,
                             q_options: QuantumInstance | Options,
                             runtime: bool,
                             runtime_service: QiskitRuntimeService | None,
                             save_checkpoint: bool,
                             start_from_checkpoint: bool,
                             directory_name: str,
                             ) -> np.ndarray:
        '''
        '''
        # Retriving the circuits, coefficients and parameters.
        (qcs, coefs, T_param) = LinearSpectroscopyCircuit(self.system)

        # Creating the measurement operator (X+iY on the ancilla qubit).
        measurement_op = PauliSumOp(SparsePauliOp.from_list([('X'+'I'*self.system.system_size, 1.),
                                                             ('Y'+'I'*self.system.system_size, 1.j),
                                                             ],
                                                            ),
                                    )

        if not runtime:
            # Initializing the response_function array and setting the number of already_examined_circuits.
            if start_from_checkpoint:
                response_function = np.load(os.path.join(directory_name, "response_function.npy"))
                already_examined_circuits = np.load(os.path.join(directory_name, "examined_circuits"))[0]

            else:
                response_function = np.zeros(len(self.delay_time), dtype='complex128')
                already_examined_circuits = 0

            # Printing a statement with the number of total circuits to be evaluated.
            tot_circuits = 2 * len(qcs) * len(self.delay_time)
            print("Total number of circuits = {}".format(tot_circuits))

            # Running the circuits.
            for n_qc, qc in enumerate(qcs):
                solved_circuits = 2 * (n_qc + 1) * len(self.delay_time)

                if solved_circuits > already_examined_circuits:
                    measurable_expression = StateFn(measurement_op, is_measurement=True).compose(CircuitStateFn(qc))
                    expectation = PauliExpectation().convert(measurable_expression)
                    sampler = CircuitSampler(q_options).convert(expectation, {T_param:self.delay_time})
                    results = np.array(sampler.eval())
                    response_function = response_function + results*coefs[n_qc]

                    # Printing a statement with the number of circuits evaluated.
                    print("Solved: {}/{}".format(solved_circuits, tot_circuits))

                    # Saving checkpoints if save_checkpoint is True.
                    if save_checkpoint:
                        np.save(os.path.join(directory_name, "response_function"), response_function)
                        np.save(os.path.join(directory_name, "examined_circuits"), np.array([solved_circuits, tot_circuits]))

        else:
            # Binding the parameters.
            qcs_job = [qc_time.bind_parameters({T_param:T})
                       for T in self.delay_time
                       for qc_time in qcs
                       ]

            # Running the circuits.
            with Session(service = runtime_service,
                         backend = self.computational_details["Backend"],
                         ):
                estimator = Estimator(options = q_options)
                runtime_job = estimator.run(circuits = qcs_job,
                                            observables = [measurement_op] * len(self.delay_time) * self.system.system_size ** 2,
                                            shots = self.computational_details["Shots"],
                                            )
                
            response_function = np.sum(np.reshape(runtime_job.result().values,
                                                  (-1, len(coefs))
                                                  ) * coefs,
                                       axis = 1,
                                       )
        
        response_function = 1.j * response_function

        return response_function

    def __ThirdOrderSpectroscopy(self,
                                 q_options: QuantumInstance | Options,
                                 runtime: bool,
                                 runtime_service: QiskitRuntimeService | None,
                                 save_checkpoint: bool,
                                 start_from_checkpoint: bool,
                                 directory_name: str,
                                 ) -> np.ndarray:
        '''
        '''
        # Retriving the circuits, coefficients and parameters.
        (qcs, coefs, T1_param, T2_param, T3_param) = ThirdOrderSpectroscopyCircuit(self.FD_type, self.system)

        # Creating the measurement operator (X+iY on the ancilla qubit).
        measurement_op = PauliSumOp(SparsePauliOp.from_list([('X'+'I'*self.system.system_size, 1.),
                                                             ('Y'+'I'*self.system.system_size, 1.j),
                                                             ],
                                                            ),
                                    )

        if not runtime:
            # Initializing the response_function array and setting the number of already_examined_circuits.
            if start_from_checkpoint:
                response_function = np.load(os.path.join(directory_name, "response_function.npy"))
                already_examined_circuits = np.load(os.path.join(directory_name, "examined_circuits"))[0]
                
            else:
                response_function = np.zeros((len(self.delay_time[0]),
                                              len(self.delay_time[1]),
                                              len(self.delay_time[2]),
                                              ),
                                             dtype ='complex128',
                                             )
                already_examined_circuits = 0

            # Creating the time combinations.
            times = [[t1,t2,t3]
                     for t1 in self.delay_time[0]
                     for t2 in self.delay_time[1]
                     for t3 in self.delay_time[2]
                     ]
            times_1 = [time[0] for time in times]
            times_2 = [time[1] for time in times]
            times_3 = [time[2] for time in times]

            # Printing a statement with the number of total circuits to be evaluated.
            tot_circuits = 2 * len(qcs) * len(self.delay_time[0]) * len(self.delay_time[1]) * len(self.delay_time[2])
            print("Total number of circuits = {}".format(tot_circuits))

            # Running the circuits.
            for n_qc, qc in enumerate(qcs):
                solved_circuits = 2 * (n_qc+1) * len(self.delay_time[0]) * len(self.delay_time[1]) * len(self.delay_time[2])
                
                if solved_circuits > already_examined_circuits:
                    measurable_expression = StateFn(measurement_op, is_measurement=True).compose(CircuitStateFn(qc))
                    expectation = PauliExpectation().convert(measurable_expression)
                    sampler = CircuitSampler(q_options).convert(expectation,
                                                                {T1_param: times_1,
                                                                T2_param: times_2,
                                                                T3_param:times_3,
                                                                },
                                                                )
                    results = np.array(sampler.eval())
                    response_function = response_function + coefs[n_qc] * np.reshape(results, (len(self.delay_time[0]), len(self.delay_time[1]), len(self.delay_time[2])))

                    # Printing a statement with the number of circuits evaluated.
                    print("Solved: {}/{}".format(solved_circuits, tot_circuits))

                    # Saving checkpoints if save_checkpoint is True.
                    if save_checkpoint:
                        np.save(os.path.join(directory_name, "response_function"), response_function)
                        np.save(os.path.join(directory_name, "examined_circuits"), np.array([solved_circuits, tot_circuits]))

        else:
            # Binding the parameters.
            qcs_job = [qc_time.bind_parameters({T1_param:T1, T2_param:T2, T3_param:T3})
                       for T1 in self.delay_time[0]
                       for T2 in self.delay_time[1]
                       for T3 in self.delay_time[2]
                       for qc_time in qcs
                       ]
            
            # Running the circuits.
            with Session(service = runtime_service,
                         backend = self.computational_details["Backend"],
                         ):
                estimator = Estimator(options = q_options)
                runtime_job = estimator.run(circuits = qcs_job,
                                            observables = [measurement_op] * len(self.delay_time[0]) * len(self.delay_time[1]) * len(self.delay_time[2]) * (2 * self.system.system_size ** 4),
                                            shots = self.computational_details["Shots"],
                                            )
            response_function = np.reshape(np.sum(np.reshape(runtime_job.result().values,
                                                             (-1, len(coefs))
                                                             )*coefs,
                                                  axis = 1,
                                                  ),
                                           (len(self.delay_time[0]),
                                            len(self.delay_time[1]),
                                            len(self.delay_time[2]),
                                            ),
                                           )
        
        if self.FD_type in ['esa', 'excited state absorption']:
            response_function = - response_function

        response_function = - 1.j * response_function

        return response_function

    def run(self,
            backend: IBMBackend | QasmSimulator,
            shots: int = 4000,
            noise_model: NoiseModel | None = None,
            coupling_map: Any = None,
            basis_gates: Any = None,
            tags: Any = None,
            runtime: bool = False,
            runtime_service: QiskitRuntimeService | None = None,
            save_Qjob: bool = True,
            save_name: str | None = None,
            save_checkpoint: bool = False,
            start_from_checkpoint: bool = False,
            **kwargs,
            ) -> np.ndarray:
        '''Run the simulation on the selected IBM Quantum backend.

        Input:
        - backend
            IBM Quantum backend.
        - shots: int
            Number of shots for the measure.
        - noise_model: qiskit_aer.noise.NoiseModel
            Noise model to be used. Can be used when backend is the qasm_simulator. To use the noise model of an existing backend use: qiskit_aer.noise.NoiseModel.from_backend(IBMQ_backend).
        - coupling_map
            Compling map of the quantum processor.
        - basis_gates
            List of basis gate names.
        - tags
            Tags to identify Qjob object.
        - runtime: bool
            If True, it uses IBM Runtime service.
        - runtime_service: QiskitRuntimeService
            The QiskitRuntimeService to be used.
        - save_Qjob: bool
            If True, it saves the information about the Qjob as a .pkl file.
        - save_name: str
            Name for the saving option.
        - save_checkpoint: bool
            If True, it generates a folder with checkpoint data. Not available when Qiskit Runtime is used.
        - start_from_checkpoint: bool
            If True, it continues an existing checkpoint. Not available when Qiskit Runtime is used.
        - kwargs
            Extra input for a qiskit.utils.QuantumInstance (if runtime is False) or qiskit_ibm_runtime.Options (if runtime is True).

        Return:
        numpy.ndarray.
        '''
        try:
            # Checking if calculation starts from an existing checkpoint.
            if start_from_checkpoint and not runtime:
                directory_name = "checkpoint_" + save_name
                path = os.path.join(os.getcwd(), directory_name)
                if not os.path.exists(path):
                    raise Exception("Checkpoint folder not found.")
            # Creating the checkpoint directory when needed.
            elif save_checkpoint and not runtime:
                if save_name == None:
                    directory_name = "checkpoint_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
                else:
                    directory_name = "checkpoint_" + save_name
                path = os.path.join(os.getcwd(), directory_name)
                existing_directory = os.path.exists(path)
                c = 1
                while existing_directory == True:
                    directory_name_c = directory_name + str(c)
                    c += 1
                    path = os.path.join(os.getcwd(), directory_name_c)
                    existing_directory = os.path.exists(path)
                os.mkdir(directory_name)
            else:
                directory_name = None

            # Saving starting time.
            starting_time = datetime.now()
            self.computational_details["Starting Time"] = starting_time.strftime("%d/%m/%Y %H:%M:%S")
            
            # Saving parameters of the quantum computation.
            self.computational_details["Backend"] = backend
            self.computational_details["Shots"] = shots
            self.computational_details["Tags"] = tags
            try:
                backend_name = backend.name()
            except:
                backend_name = backend.name
            if backend_name == "qasm_simulator" or backend_name == "ibmq_qasm_simulator":
                self.computational_details["Noise Model"] = noise_model
                self.computational_details["Coupling Map"] = coupling_map
                self.computational_details["Basis Gates"] = basis_gates
            else:
                self.computational_details["Noise Model"] = NoiseModel.from_backend(backend)
                self.computational_details["Coupling Map"] = backend.configuration().coupling_map
                self.computational_details["Basis Gates"] = NoiseModel.from_backend(backend).basis_gates
            
            # Generating the QuantumInstance or the Options
            if not runtime:
                q_options = self.__QuantumInstance_generator(**kwargs)
            elif runtime:
                q_options = self.__Options_generator(**kwargs)

            # Selecting the Feynman Diagram and solving the circuits.
            if self.FD_type in self._linear_name_list:
                self.response_function = self.__LinearSpectroscopy(q_options, runtime, runtime_service, save_checkpoint, start_from_checkpoint, directory_name)
            else:
                self.response_function = self.__ThirdOrderSpectroscopy(q_options, runtime, runtime_service, save_checkpoint, start_from_checkpoint, directory_name)

            # Saving ending time and getting the duration.
            ending_time = datetime.now()
            self.computational_details["Ending Time"] = ending_time.strftime("%d/%m/%Y %H:%M:%S")
            self.computational_details["Computation Time"] = ending_time - starting_time

            # Saving the Qjob if save is True.
            if save_Qjob and not runtime:
                if save_name == None:
                    save_name = "Qjob_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
                self.save(save_name)
            
            # Deleting the checkpoint directory and its content if save_checkpoint is True.
            if save_checkpoint and not runtime:
                shutil.rmtree(directory_name)

            return self.response_function
        
        except Exception as err:
            print("Error during computation: ", err)

            # Resetting the Qjob.
            self.computational_details = self.__reset_computational_details()
