![Alt text](Image/qudofedi_logo.png?raw=true "Title")
# QuDoFeDi
QuDoFeDi (Quantum simulation of Double-sided Feynman Diagrams) is a Python package that can be used for the computation of the optical response of exciton systems to linear and non-linear electronic spectroscopies.

QuDoFeDi is based on response theory and allows for the simulation of the double-sided Feynman diagrams that compose the response function of the system.

QuDoFeDi is compatible with IBM Qiskit and can run circuits on IBM Quantum devices.

## Installing the QuDoFeDi package and dependences
For the use of this software the follwing packages have to be installed:

- [python3](https://www.python.org/)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [qiskit](https://qiskit.org/)
- [qutip](https://qutip.org/)

We advise the use of [Conda](https://www.anaconda.com/products/individual) environments for a clean setup.

Once Conda is installed, create a new environment and switch to it by running:
```
conda create --name qudofedi_env
conda activate qudofedi_env
```

Clone the repository using `git clone`.

From the parent repository folder, in your `qudofedi_env`, the requirements can be installed with:
```
pip install -r requirements
```

Then, run the following to install the QuDoFeDi package:
```
pip install -e .
```

## Example notebooks
There are example Jupyter notebooks illustrating usage of the package in the `Examples` folder.
