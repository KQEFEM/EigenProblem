# FENicSEigenProblem

`FENicSEigenProblem` is a Python class that solves eigenvalue problems for electromagnetic systems using the finite element method (FEM) with the FEniCS library. It is designed for solving problems such as the Maxwell equations:

$$
\nabla \times \nabla \times E = \omega^2 \varepsilon \mu E
$$

with Dirichlet boundary conditions:

$$
\nabla \times E = 0
$$

.

## Features
- **Finite Element Mesh Generation**: Supports different domain types, such as `cube` and `rectangle`.
- **Eigenvalue Problem Solver**: Uses SLEPc (a suite for solving eigenvalue problems) to compute eigenvalues of the system.
- **Boundary Conditions**: Implements Dirichlet boundary conditions.
- **Test Problem Mode**: Supports predefined test problems for debugging and testing purposes.
- **Flexible Mesh Resolution**: Allows users to specify the number of nodes in the mesh and domain size.

## Requirements
- **Python 3.x**
- **FEniCS**: For the finite element method (FEM) solver.
- **SLEPc**: For solving eigenvalue problems.
- **PETSc**: A toolkit for efficient numerical computation, used by FEniCS for matrix operations.
- **pyvista (Optional)**: For visualizing the solution.
- **mpi4py**: For parallel computing support.



## Tutorial Waveguide
https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_half_loaded_waveguide.html

https://bleyerj.github.io/comet-fenicsx/intro/hyperelasticity/hyperelasticity.html
