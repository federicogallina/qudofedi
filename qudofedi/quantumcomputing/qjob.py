from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import CircuitSampler, CircuitStateFn, StateFn, PauliExpectation, PauliSumOp
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Estimator
from qiskit.utils import QuantumInstance
from qiskit_aer.noise import NoiseModel
import numpy as np
import pickle
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
    '''Load the information about a Qjob from a .pkl file.
    
    Input:
    - name: str
        Name of the origin file.
    '''
    if not name.endswith(".pkl"):
        name = name + ".pkl"

    with open(name, "rb") as f:
        qjob_dict = pickle.load(f)
    
    job = Qjob().from_dict(qjob_dict)

    return job

class Qjob():
    _linear_name_list = ["a", "absorption"]
    _thirdorder_name_list = ["gsb", "ground state bleaching", "se", "stimulated emission", "esa", "excited state absorption"]

    def __reset_computational_details(self,
                                      ):
        self.computational_details = {"Backend": None,
                                      "Shots": None,
                                      "Noise Model": None,
                                      "Coupling Map": None,
                                      "Basis Gates": None,
                                      "Tags": None,
                                      "Starting Time": None,
                                      "Ending Time": None,
                                      "Computation Time": None,
                                      "Results per Circuit": None,
                                      }

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
        self.__reset_computational_details()
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
        #Retriving the circuits.
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
        #Retriving the circuits.
        qcs = self.get_circuits(self.system)
        
        #Checking what to print
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
        qjob_dict = {"System Size":self.system.system_size,
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
        self.FD_type = qjob_dict["Feynman Diagram Type"]
        self.delay_time = qjob_dict["Delay Time"]
        self.system = System(qjob_dict["System Hamiltonian"],
                            qjob_dict["System Dipole Moment Amplitudes"],
                            )
        self.computational_details = qjob_dict["Computational details"]
        self.response_function = qjob_dict["Response Function"]

        return self

    def save(self,
             name: str,
             ):
        '''Save the information about the Qjob as a .pkl file.
        
        Input:
        - name: str
            Name of the destination file.
        '''
        if not name.endswith(".pkl"):
            name = name + ".pkl"

        with open(name, "wb") as f:
            pickle.dump(self.to_dict(), f)
        
        print("File saved in "
              + os.getcwd()
              + name)

    def load(self,
             name: str,
             ):
        '''Load the information about a Qjob from a .pkl file.
        
        Input:
        - name: str
            Name of the origin file.
        '''
        if not name.endswith(".pkl"):
            name = name + ".pkl"

        with open(name, "rb") as f:
            qjob_dict = pickle.load(f)
        
        self.from_dict(qjob_dict)

        return self

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
                             directory_name: str,
                             ) -> np.ndarray:
        '''
        '''
        #Retriving the circuits, coefficients and parameters.
        (qcs, coefs, T_param) = LinearSpectroscopyCircuit(self.system)

        #Creating the measurement operator (X+iY on the ancilla qubit).
        measurement_op = PauliSumOp(SparsePauliOp.from_list([('X'+'I'*self.system.system_size, 1.),
                                                             ('Y'+'I'*self.system.system_size, 1.j),
                                                             ],
                                                            ),
                                    )

        if not runtime:
            #Initializing the output array.
            response_function = np.zeros(len(self.delay_time), dtype='complex128')

            #Running the circuits.
            for n_qc, qc in enumerate(qcs):
                measurable_expression = StateFn(measurement_op, is_measurement=True).compose(CircuitStateFn(qc))
                expectation = PauliExpectation().convert(measurable_expression)
                sampler = CircuitSampler(q_options).convert(expectation, {T_param:self.delay_time})
                results = np.array(sampler.eval())
                response_function = response_function + results*coefs[n_qc]

                #Saving checkpoints if save_checkpoint is True.
                if save_checkpoint:
                    np.save(os.path.join(directory_name, "response_function"), response_function)
                    np.save(os.path.join(directory_name, "examined_circuits"), np.array([n_qc + 1, len(qcs)]))

        else:
            #Binding the parameters.
            qcs_job = [qc_time.bind_parameters({T_param:T})
                       for T in self.delay_time
                       for qc_time in qcs
                       ]

            #Running the circuits.
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

        return response_function

    def __ThirdOrderSpectroscopy(self,
                                 q_options: QuantumInstance | Options,
                                 runtime: bool,
                                 runtime_service: QiskitRuntimeService | None,
                                 save_checkpoint: bool,
                                 directory_name: str,
                                 ) -> np.ndarray:
        '''
        '''
        #Retriving the circuits, coefficients and parameters.
        (qcs, coefs, T1_param, T2_param, T3_param) = ThirdOrderSpectroscopyCircuit(self.FD_type, self.system)

        #Creating the measurement operator (X+iY on the ancilla qubit).
        measurement_op = PauliSumOp(SparsePauliOp.from_list([('X'+'I'*self.system.system_size, 1.),
                                                             ('Y'+'I'*self.system.system_size, 1.j),
                                                             ],
                                                            ),
                                    )

        if not runtime:
            #Initializing the output array.
            response_function = np.zeros((len(self.delay_time[0]),
                                          len(self.delay_time[1]),
                                          len(self.delay_time[2]),
                                          ),
                                         dtype ='complex128',
                                         )

            #Creating the time combinations.
            times = [[t1,t2,t3]
                     for t1 in self.delay_time[0]
                     for t2 in self.delay_time[1]
                     for t3 in self.delay_time[2]
                     ]
            times_1 = [time[0] for time in times]
            times_2 = [time[1] for time in times]
            times_3 = [time[2] for time in times]

            #Printing a statement with the number of total circuits to be evaluated.
            tot_circuits = len(qcs) * len(self.delay_time[0]) * len(self.delay_time[1]) * len(self.delay_time[2])
            print("Total number of circuits = {}".format(tot_circuits))

            #Running the circuits.
            for n_qc, qc in enumerate(qcs):
                measurable_expression = StateFn(measurement_op, is_measurement=True).compose(CircuitStateFn(qc))
                expectation = PauliExpectation().convert(measurable_expression)
                sampler = CircuitSampler(q_options).convert(expectation, {T1_param:times_1, T2_param:times_2, T3_param:times_3})
                results = np.array(sampler.eval())
                response_function = response_function + coefs[n_qc] * np.reshape(results, (len(self.delay_time[0]), len(self.delay_time[1]), len(self.delay_time[2])))

                #Printing a statement with the number of circuits evaluated.
                solved_circuits = (n_qc+1) * len(self.delay_time[0]) * len(self.delay_time[1]) * len(self.delay_time[2])
                print("Solved: {}/{}".format(solved_circuits, tot_circuits))

                #Saving checkpoints if save_checkpoint is True.
                if save_checkpoint:
                    np.save(os.path.join(directory_name, "response_function"), response_function)
                    np.save(os.path.join(directory_name, "examined_circuits"), np.array([solved_circuits, tot_circuits]))
        else:
            #Binding the parameters.
            qcs_job = [qc_time.bind_parameters({T1_param:T1, T2_param:T2, T3_param:T3})
                       for T1 in self.delay_time[0]
                       for T2 in self.delay_time[1]
                       for T3 in self.delay_time[2]
                       for qc_time in qcs
                       ]
            
            #Running the circuits.
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
        return response_function

    def run(self,
            backend,
            shots: int = 4000,
            noise_model: NoiseModel | None = None,
            coupling_map: Any = None,
            basis_gates: Any = None,
            tags: Any = None,
            runtime: bool = False,
            runtime_service: QiskitRuntimeService | None = None,
            save_Qjob: bool = False,
            save_name: str | None = None,
            save_checkpoint: bool = False,
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
            If True uses IBM Runtime service.
        - runtime_service: QiskitRuntimeService
            The QiskitRuntimeService to be used.
        - save_Qjob: bool
            If True save the information about the Qjob as a .pkl file.
        - save_name: str
            Name of the destination file.
        - save_checkpoint: bool
            If True generates a folder with checkpoint data.
        - kwargs
            Extra input for a qiskit.utils.QuantumInstance (if runtime is False) or qiskit_ibm_runtime.Options (if runtime is True).

        Return:
        numpy.ndarray.
        '''
        try:
            #Creating the checkpoint directory if save_checkpoint is True.
            if save_checkpoint and not runtime:
                if save_name == None:
                    directory_name = "checkpoint_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
                else:
                    directory_name = "checkpoint_" + save_name
                os.mkdir(directory_name)
            else:
                directory_name = None

            #Saving starting time.
            starting_time = datetime.now()
            self.computational_details["Starting Time"] = starting_time.strftime("%d/%m/%Y %H:%M:%S")
            
            #Saving parameters of the quantum computation.
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
            
            #Generating the QuantumInstance or the Options
            if not runtime:
                q_options = self.__QuantumInstance_generator(**kwargs)
            elif runtime:
                q_options = self.__Options_generator(**kwargs)

            #Selecting the Feynman Diagram and solving the circuits.
            if self.FD_type in self._linear_name_list:
                self.response_function = self.__LinearSpectroscopy(q_options, runtime, runtime_service, save_checkpoint, directory_name)
            else:
                self.response_function = self.__ThirdOrderSpectroscopy(q_options, runtime, runtime_service, save_checkpoint, directory_name)

            #Saving ending time and getting the duration.
            ending_time = datetime.now()
            self.computational_details["Ending Time"] = ending_time.strftime("%d/%m/%Y %H:%M:%S")
            self.computational_details["Computation Time"] = ending_time - starting_time

            #Saving the Qjob if save is True.
            if save_Qjob and not runtime:
                if save_name == None:
                    save_name = "Qjob_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
                self.save(save_name)
            
            #Deleting the checkpoint directory and its content if save_checkpoint is True.
            if save_checkpoint and not runtime:
                shutil.rmtree(directory_name)

            return self.response_function
        
        except Exception as err:
            print("Error during computation: ", err)

            #Resetting the Qjob.
            self.__reset_computational_details()

            #Deleting the checkpoint directory and its content if save_checkpoint is True.
            if save_checkpoint and not runtime:
                shutil.rmtree(directory_name)
