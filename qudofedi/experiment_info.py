import numpy as np
from qutip import Qobj
from typing import Literal

class System():
    def __init__(self,
                 Hamiltonian: list[list[float]] | np.ndarray | Qobj,
                 dipole_moment_amplitude: list[float] | list[int] | float | int = [1],
                 ):
        '''Create an object that contains the information about the system intended as a collection of (possibly) interactive two-level electronic systems.

        Input:
        - Hamiltonian: Union[list[list[float]], numpy.ndarray, qutip.Qobj]
            The electronic Hamiltonian of the system on the site basis.
        - dipole_moment_amplitude: Union[list[float], list[int], float, int]
            A list of the amplitude of the dipole moments corresponding to the two-level systems that compose the whole system. If a monomer system is considered a float is accepted as input.
        '''
        #Checking the Hermiticity of the Hamiltonian. Converting Hamiltonian to numpy.ndarray.
        if isinstance(Hamiltonian, Qobj):
            if Hamiltonian.isherm == False:
                raise Exception('Hamiltonian is not Hermitian.')
            Hamiltonian = Hamiltonian.full()
        else:
            Hamiltonian = np.array(Hamiltonian)
            if Qobj(Hamiltonian).isherm == False:
                raise Exception('Hamiltonian is not Hermitian.')
            
        #Checking the shape of the Hamiltonian.
        if Hamiltonian.ndim < 2 or Hamiltonian.ndim > 2 or Hamiltonian.shape[0] != Hamiltonian.shape[1]:
            raise Exception('Hamiltonian is not a square matrix.')
        
        #Checking that the size of the Hamiltonian is a power of 2.
        if (Hamiltonian.shape[0] & Hamiltonian.shape[0]-1) != 0:
            raise Exception('Hamiltonian dimension is {}. However, it must be a power of 2.'.format(Hamiltonian.shape[0]))
        
        #Saving information about the Hamiltonian and system size.
        self.Hamiltonian = Hamiltonian
        self.system_size = int(np.log2(Hamiltonian.shape[0]))

        #Checking the number of dipole moments.
        if isinstance(dipole_moment_amplitude, float) or isinstance(dipole_moment_amplitude, int):
            dipole_moment_amplitude = [dipole_moment_amplitude]
        if not isinstance(dipole_moment_amplitude, list):
            raise TypeError('dipole_moment_amplitude is a {}. However, for multi-chromophoric systems dipole_moment_amplitude must be given as a list of floats. While for a single chromophore a float is also accepted.'.format(type(dipole_moment_amplitude)))
        if len(dipole_moment_amplitude) != self.system_size:
            raise Exception('The number of dipole moments is {}, which is in conflict with the system size (number of chromophores = {}).'.format(len(dipole_moment_amplitude), self.system_size))
        
        #Saving information about the dipole moments.
        self.dipole_moment = dipole_moment_amplitude

class FeynmanDiagram():
    _linear_name_list = ["a", "absorption"]
    _thirdorder_name_list = ["gsb", "ground state bleaching", "se", "stimulated emission", "esa", "excited state absorption"]

    def __init__(self,
                 FD_type: Literal["a", "absorption", "gsb", "ground state bleaching", "se", "stimulated emission", "esa", "excited state absorption"],
                 delay_time: np.ndarray | list[list[float]] | list[float] | float | list[list[int]] | list[int] | int,
                 ):
        '''Create an object that contains the information about the contribution of the response function to be simulated.

        Input:
        - FD_type: Literal["a", "absorption", "gsb", "ground state bleaching", "se", "stimulated emission", "esa", "excited state absorption"]
            The name of the Feynman diagram. At the moment, only linear absorption and the components of the third-order rephasing signal are implemented.
        - delay_time: Union[numpy.ndarray, list[list[float]], list[float], float, list[list[int]], list[int], int]
            A list with the delay times. For the linear absorption the input is: list[float] or float. For third-order responses the input is a list with 3 entres (i.e.: [T1_list, T2_list, T3_list]), therefore the accepted types are: list[list[float]] or list[float].
        '''
        #Checking the correct input format for delay_time. Converting float to list when necessary.
        if FD_type.casefold() in self._linear_name_list:
            if isinstance(delay_time, float) or isinstance(delay_time, int):
                delay_time = [delay_time]
            elif isinstance(delay_time, np.ndarray):
                delay_time = delay_time.tolist()
            elif not isinstance(delay_time, list):
                raise ValueError('delay_time is a {}. However, a list, numpy.ndarray or a float are expected.'.format(type(delay_time)))
        elif FD_type in self._thirdorder_name_list:
            if not isinstance(delay_time, list):
                raise ValueError('delay_time is a {}. However, a list is expected.'.format(type(delay_time)))
            if len(delay_time) != 3:
                raise ValueError('delay_time contains {} elements. However, 3 are expected.'.format(len(delay_time)))
            for i,T_i in enumerate(delay_time):
                if isinstance(T_i, float) or isinstance(T_i, int):
                    delay_time[i] = [T_i]
                elif isinstance(T_i, np.ndarray):
                    delay_time[i] = T_i.tolist()
                elif not isinstance(T_i, list):
                    raise ValueError('delay_time[{}] is a {}. However, a list, numpy.ndarray or a float are expected.'.format(i, type(T_i)))
                
        #Checking that the name of the Feynman diagram is correct.
        else:
            raise ValueError('Invalid name of the Feynman diagram')
        
        #Saving type of Feynamn diagram and the list of delay times.
        self.type = FD_type.casefold()
        self.delay_time = delay_time
