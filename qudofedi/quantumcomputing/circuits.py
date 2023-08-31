from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import HamiltonianGate
from qiskit.circuit import Parameter
from qiskit.circuit.library import RZGate
from typing import Literal
from qudofedi.experiment_info import System


def LinearSpectroscopyCircuit(system: System,
                              ) -> tuple[list[QuantumCircuit], list[float], Parameter]:
    '''Create the parametrized circuit for the Feynman Diagrams corresponding to the Linear Response Function (Absorption).
    Delay time is the parameter.

    Input:
    - system: System
        The system of interest.
    '''
    #Generating the parameter.
    T_param = Parameter("T")

    #Generating the evolution gate. If the system is a monomer, a simplified RZ gate is used instead of an Hamiltonian Gate.
    #Further techniques (such as Trotterization) can be implemented in this part of the code to deal with large systems.
    if system.system_size == 1:
        energy_gap = system.Hamiltonian[1,1] - system.Hamiltonian[0,0]
        U = RZGate(- energy_gap * T_param)
    else:
        U = HamiltonianGate(system.Hamiltonian, T_param)

    #Generating the circuit with the ancilla in superposition.
    sys_reg = QuantumRegister(system.system_size, "system")
    anc_reg = QuantumRegister(1, "ancilla")
    initial_qc = QuantumCircuit(sys_reg, anc_reg)
    initial_qc.h(anc_reg[0])

    #Creating the lists for the circuits and the coefficients.
    qcs = []
    coefs = []

    #Generating the parametrized circuits for the Feynman Diagrams.
    for i in range(system.system_size):
        coef_i = system.dipole_moment[i]
        for j in range(system.system_size):
            coef_j = system.dipole_moment[j]

            #Taking the product of the dipole moments and saving it in the coefs list.
            coef = coef_i * coef_j
            coefs.append(coef)

            #Creating the circuit and saving it in the qcs list.
            qc = initial_qc.copy()
            qc.cx(anc_reg[0], sys_reg[system.system_size - i - 1])
            qc.append(U, sys_reg)
            qc.cx(anc_reg[0], sys_reg[system.system_size - j - 1])
            qcs.append(qc)

    return qcs, coefs, T_param

def ThirdOrderSpectroscopyCircuit(FD_type: Literal["gsb", "ground state bleaching", "se", "stimulated emission", "esa", "excited state absorption"],
                                  system: System,
                                  ) -> tuple[list[QuantumCircuit], list[float], Parameter, Parameter, Parameter]:
    '''Create the parametrized circuit for the Feynman Diagrams corresponding to the Third-Order(-Rephasing) Response Function (GSB, SE, ESA).
    Delay times (T1, T2, T3) are the parameters.
    Note: we assume that the system starts in its ground state. Therefore, the first dipol operator acting on the bra and ket side of the density matrix can be mu instead of mu^- or mu^+.

    Input:
    - FD_type: Literal["gsb", "ground state bleaching", "se", "stimulated emission", "esa", "excited state absorption"]
        The name of the Feynman diagram. At the moment, only linear absorption and the components of the third-order rephasing signal are implemented.
    - system: System
        The system of interest.
    '''
    #Generating the parameters.
    T1_param = Parameter("T1")
    T2_param = Parameter("T2")
    T3_param = Parameter("T3")

    #Generating the circuits. If the system is a monomer, a simplified generator is used.
    if system.system_size == 1:
        (qcs, coefs) = __TOSC_TLS(FD_type, T1_param, T2_param, T3_param, system)
    
    else:
        #Generating the evolution gate.
        #Further techniques (such as Trotterization) can be implemented in this part of the code to deal with large systems.
        U1 = HamiltonianGate(system.Hamiltonian, T1_param)
        U2 = HamiltonianGate(system.Hamiltonian, T2_param)
        U3 = HamiltonianGate(system.Hamiltonian, T3_param)

        #Generating the circuit with the ancilla in superposition.
        sys_reg = QuantumRegister(system.system_size, "system")
        anc_reg = QuantumRegister(1, "ancilla")
        initial_qc = QuantumCircuit(sys_reg, anc_reg)
        initial_qc.h(anc_reg[0])

        #Creating the lists for the circuits and the coefficients.
        qcs = []
        coefs = []

        #Generating the parametrized circuits for the GSB Feynman Diagrams.
        if FD_type == "gsb" or FD_type == "ground state bleaching":
            for i in range(system.system_size):
                coef_i = system.dipole_moment[i] #mu (since starting from the ground state. Otherwise: mu^-).
                for j in range(2*system.system_size):
                    coef_j = system.dipole_moment[j]/2 if j<system.system_size else -1.j/2*system.dipole_moment[j-system.system_size] #mu^+.
                    for k in range(system.system_size):
                        coef_k = system.dipole_moment[k] #mu (since starting from the ground state. Otherwise: mu^+).
                        for l in range(system.system_size):
                            coef_l = system.dipole_moment[l] #mu (since we use TLS. Otherwise: mu^-).

                            #Taking the product of the dipole moments and saving it in the coefs list.
                            coef = coef_i*coef_j*coef_k*coef_l
                            coefs.append(coef)

                            #Creating the circuit and saving it in the qcs list.
                            qc = initial_qc.copy()
                            qc.cx(anc_reg[0],sys_reg[system.system_size-i-1], ctrl_state = 0)
                            qc.append(U1,sys_reg)
                            qc.cx(anc_reg[0],sys_reg[system.system_size-j-1], ctrl_state = 0) if j<system.system_size else qc.cy(anc_reg[0],sys_reg[2*system.system_size-j-1], ctrl_state = 0)
                            qc.append(U2,sys_reg)
                            qc.cx(anc_reg[0],sys_reg[system.system_size-k-1])
                            qc.append(U3,sys_reg)
                            qc.cx(anc_reg[0],sys_reg[system.system_size-l-1])
                            qcs.append(qc)

        #Generating the parametrized circuits for the SE Feynman Diagrams.
        elif FD_type == 'se' or FD_type == 'stimulated emission':
            for i in range(system.system_size):
                coef_i = system.dipole_moment[i] #mu (since starting from the ground state. Otherwise: mu^-).
                for j in range(system.system_size):
                    coef_j = system.dipole_moment[j] #mu (since starting from the ground state. Otherwise: mu^+).
                    for k in range(2*system.system_size):
                        coef_k = system.dipole_moment[k]/2 if k<system.system_size else -1.j/2*system.dipole_moment[j-system.system_size] #mu^+.
                        for l in range(system.system_size):
                            coef_l = system.dipole_moment[l] #mu (since we use TLS. Otherwise: mu^-).
                            
                            #Taking the product of the dipole moments and saving it in the coefs list.
                            coef = coef_i*coef_j*coef_k*coef_l
                            coefs.append(coef)

                            #Creating the circuit and saving it in the qcs list.
                            qc = initial_qc.copy()
                            qc.cx(anc_reg[0],sys_reg[system.system_size-i-1], ctrl_state = 0)
                            qc.append(U1,sys_reg)
                            qc.cx(anc_reg[0],sys_reg[system.system_size-j-1])
                            qc.append(U2,sys_reg)
                            qc.cx(anc_reg[0],sys_reg[system.system_size-k-1], ctrl_state = 0) if k<system.system_size else qc.cy(anc_reg[0],sys_reg[2*system.system_size-k-1], ctrl_state = 0)
                            qc.append(U3,sys_reg)
                            qc.cx(anc_reg[0],sys_reg[system.system_size-l-1])
                            qcs.append(qc)

        #Generating the parametrized circuits for the ESA Feynman Diagrams.
        elif FD_type == 'esa' or FD_type == 'excited state absorption':
            for i in range(system.system_size):
                coef_i = system.dipole_moment[i] #mu (since starting from the ground state. Otherwise: mu^-).
                for j in range(system.system_size):
                    coef_j = system.dipole_moment[j]#mu (since starting from the ground state. Otherwise: mu^+).
                    for k in range(2*system.system_size):
                        coef_k = system.dipole_moment[k]/2 if k<system.system_size else -1.j/2*system.dipole_moment[j-system.system_size] #mu^+.
                        for l in range(system.system_size):
                            coef_l = system.dipole_moment[l] #mu (since we use TLS. Otherwise: mu^-).

                            #Taking the product of the dipole moments and saving it in the coefs list.
                            coef = coef_i*coef_j*coef_k*coef_l
                            coefs.append(coef)
                            
                            #Creating the circuit and saving it in the qcs list.
                            qc = initial_qc.copy()
                            qc.cx(anc_reg[0],sys_reg[system.system_size-i-1], ctrl_state = 0)
                            qc.append(U1,sys_reg)
                            qc.cx(anc_reg[0],sys_reg[system.system_size-j-1])
                            qc.append(U2,sys_reg)
                            qc.cx(anc_reg[0],sys_reg[system.system_size-k-1]) if k<system.system_size else qc.cy(anc_reg[0],sys_reg[2*system.system_size-k-1])
                            qc.append(U3,sys_reg)
                            qc.cx(anc_reg[0],sys_reg[system.system_size-l-1])
                            qcs.append(qc)
                        
    return qcs, coefs, T1_param, T2_param, T3_param

def __TOSC_TLS(FD_type: Literal["gsb", "ground state bleaching", "se", "stimulated emission", "esa", "excited state absorption"],
                                        T1_param: Parameter,
                                        T2_param: Parameter,
                                        T3_param: Parameter,
                                        system: System,
                                        ):
    '''Just a faster routine for the monomer case (two-level system)'''

    #Generating the evolution gate. Since the system is a monomer, simplified RZ gates are used instead of an Hamiltonian Gate.
    energy_gap = system.Hamiltonian[1,1] - system.Hamiltonian[0,0]
    U1 = RZGate(- energy_gap * T1_param)
    U2 = RZGate(- energy_gap * T2_param)
    U3 = RZGate(- energy_gap * T3_param)

    #Generating the circuit with the ancilla in superposition.
    sys_reg = QuantumRegister(system.system_size, "system")
    anc_reg = QuantumRegister(1, "ancilla")
    qc = QuantumCircuit(sys_reg, anc_reg)
    qc.h(anc_reg[0])

    #Generating the parametrized circuits.
    qc.cx(1, 0, ctrl_state = 0)
    qc.append(U1, sys_reg)
    if FD_type == 'gsb' or FD_type == 'ground state bleaching':
        qc.cx(1, 0, ctrl_state = 0)
        qc.append(U2, sys_reg)
        qc.cx(1, 0)
    if FD_type == 'se' or FD_type == 'stimulated emission':
        qc.cx(1, 0)
        qc.append(U2, sys_reg)
        qc.cx(1, 0, ctrl_state = 0)
    if FD_type == 'esa' or FD_type == 'excited state absorption':
        qc.cx(1, 0)
        qc.append(U2, sys_reg)
        qc.cx(1, 0)
    qc.append(U3, sys_reg)
    qc.cx(1, 0)

    #Creating the lists for the circuits and the coefficients.
    qcs = [qc]
    coefs = [system.dipole_moment[0] ** 4]
    
    return qcs, coefs

def X_measure(qc_input: QuantumCircuit,
              qubit,
              ) -> QuantumCircuit:
    '''Add a measure along X-axis on a copy of the circuit (no modifications inplace).
    
    Input:
    - qc_input: qiskit.QuantumCircuit
        The quantum circuit.
    - qubit
        The qubit that have to be measured.
        
    Return:
    QuantumCircuit
    '''
    #Creating a copy of the circuit.
    qc = qc_input.copy()

    #Adding a ClassicalRegister
    cl_reg = ClassicalRegister(1)
    qc.add_register(cl_reg)

    #Appending the X measurement
    qc.barrier()
    qc.h(qubit)
    qc.measure(qubit, cl_reg[0])

    return qc

def Y_measure(qc_input: QuantumCircuit,
              qubit,
              ) -> QuantumCircuit:
    '''Add a measure along Y-axis on a copy of the circuit (no modifications inplace).
    
    Input:
    - qc_input: qiskit.QuantumCircuit
        The quantum circuit.
    - qubit
        The qubit that have to be measured.
        
    Return:
    QuantumCircuit
    '''
    #Creating a copy of the circuit.
    qc = qc_input.copy()

    #Adding a ClassicalRegister
    cl_reg = ClassicalRegister(1)
    qc.add_register(cl_reg)

    #Appending the Y measurement
    qc.barrier()
    qc.sdg(qubit)
    qc.h(qubit)
    qc.measure(qubit, cl_reg[0])

    return qc
