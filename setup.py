from setuptools import (
    setup,
    find_packages
)

setup(
    name = 'qudofedi',
    version = 0.1,
    author = 'Federico Gallina',
    author_email = 'federico.gallina@unipd.it',
    description = 'Quantum simulation of double-sided Feynman diagrams',
    long_description = 'QuDoFeDi (Quantum simulation of Double-sided Feynman Diagrams) is a Python package \
                        that can be used for the computation of the optical response of exciton systems \
                        to linear and non-linear electronic spectroscopies. \
                        QuDoFeDi is based on response theory and allows for the simulation of the double-sided Feynman diagrams \
                        that compose the response function of the system. \
                        QuDoFeDi is compatible with IBM Qiskit and can run circuits on IBM Quantum devices.',
    packages = find_packages(),
    url = 'https://github.com/federicogallina/qudofedi',
    install_requires = ['numpy', 'scipy', 'qiskit']
)