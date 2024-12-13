"""
This script demonstrates how to use the `FENicSEigenProblem` class to solve an eigenvalue problem for an electromagnetic system using the finite element method (FEM) with the FEniCS library.

The class `FENicSEigenProblem` is designed to solve problems like the Maxwell equations for electromagnetic fields, specifically for eigenvalue problems in a given domain with Dirichlet boundary conditions.

### Parameters:
- `num_nodes` (int, optional): The number of nodes in the mesh for the finite element method (FEM) simulation. The default value is 100. A higher value results in a finer mesh and more accurate results but may increase computational time. (e.g., `num_nodes=50`).
  
- `domain_type` (str, optional): The type of domain to be used for the problem. It can be:
  - `"cube"`: A cubic domain (default is a unit cube of size 1x1x1).
  - `"rectangle"`: A rectangular domain (requires specifying two dimensions).
  - `"circle"` or `"square"`: Other domain types are possible with further extensions.
  
- `num_eigenvalues` (int, optional): The number of eigenvalues to compute. Default is 10. You can specify a different number based on your requirements, such as `num_eigenvalues=25`.

"""

import Classes.EigenProblemClass as EP

# Create an instance of the FENicSEigenProblem class
eigen_problem = EP.FENicSEigenProblem(
    num_nodes=100, domain_type="cube", num_eigenvalues=25*2
)
eigen_problem.domain = [1, 1, 1]

# Run the eigenvalue problem
eigen_problem.run()
