# FENicSEigenProblem

`FENicSEigenProblem` is a Python class that solves eigenvalue problems for electromagnetic systems using the finite element method (FEM) with the FEniCS library. It is designed for solving problems such as the Maxwell equations:

$$
\nabla \times \nabla \times E = \omega^2 \varepsilon \mu E
$$

with Dirichlet boundary conditions:

$$
\nabla \times E = 0.
$$

Currently this code only runs for a cube and dirichlet BCs but can be extended. 

## Running
To run this code use the main.py file and change the parameters are your leisure. The eigenvalues are saved in a pickle file in the folder 'data' locally.

Alternatively, use the following:
```bash
./run.sh build
```
If you want to then run it only in the terminal use 

```bash 
./run.sh run
```

If you are using vs code you could alternatively open the _devcontainer.json_ by using _Remote: Reopen in Container_. This then opens vs code as normal. 

*Note: you should use the environment fenics as all the requirements are installed there.*



## Features
- **Finite Element Mesh Generation**: Supports different domain types, such as `cube` and `rectangle`.
- **Eigenvalue Problem Solver**: Uses SLEPc (a suite for solving eigenvalue problems) to compute eigenvalues of the system.
- **Boundary Conditions**: Implements Dirichlet boundary conditions.
- **Test Problem Mode**: Supports predefined test problems for debugging and testing purposes.
- **Flexible Mesh Resolution**: Allows users to specify the number of nodes in the mesh and domain size.

## Requirements
These are provided in the requirements.txt (except for fenicsx) on the linux subsystem. (mamba create -n <env_name> --file requirements.txt)
- **Python 3.9**
- **FEniCS**: For the finite element method (FEM) solver. See [here](https://fenicsproject.org/download/) for installation instructions (recommend using mamba for speed).
- **SLEPc**: For solving eigenvalue problems.
- **PETSc**: A toolkit for efficient numerical computation, used by FEniCS for matrix operations.
- **pyvista (Optional)**: For visualizing the solution.
- **mpi4py**: For parallel computing support.

## To be Done
We can introduce new domains, and boundary conditions. 

## Tutorial Waveguide
https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_half_loaded_waveguide.html

https://bleyerj.github.io/comet-fenicsx/intro/hyperelasticity/hyperelasticity.html

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to share and adapt the material, provided you give appropriate credit and do not use it for commercial purposes.
