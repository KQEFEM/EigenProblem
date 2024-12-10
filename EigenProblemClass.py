"""
This is for solving mask deformation problems using FEniCS (FEM).
"""

import os

# from dolfinx.fem import FunctionSpace, dirichletbc, Constant, locate_dofs_topological, TrialFunction, TestFunction# import mshr  # package in fenics
import pickle
import shutil
import time

import dolfinx.fem.petsc as fem_petsc
import numpy as np
import ufl as ufl
from dolfinx import fem
from dolfinx.mesh import CellType, create_box
from mpi4py import MPI

try:
    import pyvista

    have_pyvista = True
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False

try:
    from slepc4py import SLEPc
except ModuleNotFoundError:
    print("slepc4py is required for this demo")
    exit(0)


class FENicSEigenProblem:
    """
    This solves for the eigenvalues of the maxwell problem \nabla \times \nabla \times E = \omega^2 \varepsilon \mu E for the BC of \nabla \times E = 0.

    Running the code on MACOS:
        Docker command to start fenics
        docker run -ti -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:current

    """

    def __init__(
        self,
        num_nodes: int = 100,
        domain_type: str = "cube",
        test_problem: bool = False,
        test_mode: bool = False,
        num_eigenvalues=10,
    ):
        """

          Args:
            num_nodes (int): Number of nodes for mesh generation (default: 100).
            domain_type (str): Type of domain (e.g., 'cube', 'rectangle').
            test_problem (bool): Run a predefined test problem (default: False).
            test_mode (bool): Enable testing mode for pytest (default: False).
            num_eigenvalues (int): Number of eigenvalues to compute (default: 10).
      
        """
        self.num_nodes = num_nodes
        self.domain_type = domain_type
        self.test_problem = test_problem
        self.test_mode = test_mode
        self.num_eigenvalues = num_eigenvalues

        # Set default tolerances and placeholders for FEniCS objects
        self.tol = 1e-9
        self.mesh = None
        self.V = None
        self.u = None
        self.v = None
        self.bc = None
        self.RHS = None

        # Initialize domain and boundary settings
        self.domain = []
        self.boundary_function = BoundaryFunction()
        self.boundary_def = None

        # Experiment-related metadata
        self.experiment_name = None
        self.subfolder_name = None
        self.script_dir = os.path.dirname(os.path.realpath(__file__))

    def set_constants(self):
        self.frequency = fem.Constant(self.mesh, 1.0)  # omega = 0 for unexcited system
        self.permittivity = fem.Constant(self.mesh, 1.0)  # epsilon
        self.permeability = fem.Constant(self.mesh, 1.0)  # mu

    def create_mesh_and_function_space(self):
        """Creates the mesh for the problem. This is either a rectangle or a circle."""
        if self.domain_type.lower() == "rectangle":
            self.mesh = fem.RectangleMesh(
                fem.Point(0, 0),
                fem.Point(self.domain[0], self.domain[1]),
                self.num_nodes,
                self.num_nodes,
            )
        elif self.domain_type.lower() == "cube":
            self.domain = [1, 1, 1]
            self.num_nodes = 10
            print("Domain:", self.domain)
            print("Num nodes:", self.num_nodes)
            self.mesh = create_box(
                MPI.COMM_WORLD,
                [
                    np.array([0, 0, 0]),
                    np.array([self.domain[0], self.domain[1], self.domain[2]]),
                ],
                [self.num_nodes, self.num_nodes, self.num_nodes],
                CellType.hexahedron,
            )
        else:
            print("\nStandard unit square is used.\n")
            self.mesh = fem.UnitSquareMesh(self.num_nodes, self.num_nodes)

        self.V = fem.functionspace(self.mesh, ("P", 1))
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

    def RHS_test_problem(self):
        """
        Define the right-hand side function for the equation. for u(x,y) = sin(x)sin(y), the right-hand side function is -2sin(x)sin(y)

        Parameters:
        degree (int): The degree of polynomial approximation used when the expression
                    is interpolated onto a finite element function space. It should
                    match the degree of the finite element space where this expression
                    will be used. For example, for linear or "P1" elements, use degree=1;
                    for quadratic or "P2" elements, use degree=2.
        """
        self.rhs_func = -fem.Expression("-2*sin(x[0])*sin(x[1])", degree=1)

    def weak_form(self):
        """Computes the weak form of the problem."""

        curl_curl = ufl.dot(ufl.curl(self.u), ufl.curl(self.v)) * ufl.dx
        mass = (
            self.frequency**2
            * self.permittivity
            * self.permeability
            * ufl.dot(self.u, self.v)
            * ufl.dx
        )
        self.LHS = curl_curl + mass
        self.RHS = 0

    def boundary_condition(self):
        """Create a Dirichlet boundary condition on the entire boundary.
        Parameters: V: FunctionSpace - The function space to apply the boundary condition to. value: float - The value of the Dirichlet condition (default is 0.0).
        Returns: bc: DirichletBC - The boundary condition."""
        # Define the value of the boundary condition
        value = 0.0
        u_D = fem.Constant(self.V.mesh, value)
        # Locate the degrees of freedom on the boundary
        boundary_dofs = fem.locate_dofs_geometrical(
            self.V, lambda x: np.full(x.shape[1], True)
        )
        # Create the boundary condition
        self.bc = [fem.dirichletbc(u_D, boundary_dofs, self.V)]  # Wrap in list

    def solve_eigenvalue_problem(self):
        """Solves the eigenvalue problem for the bilinear form using SLEPc."""
        # Create a solution function in the function space
        self.solution = fem.Function(self.V)

        # Define the bilinear form
        A = fem_petsc.assemble_matrix(fem.form(self.LHS), bcs=self.bc)
        A.assemble()
        self.eps = SLEPc.EPS().create(self.mesh.comm)  # This represents the solver
        self.eps.setOperators(A)  # Set the operators for the eigenvalue problem
        """ If the matrices in the problem have known properties (e.g. hermiticity) we can use this information in SLEPc to accelerate the calculation with the setProblemType function. For this problem, there is no property that can be exploited, and therefore we define it as a generalized non-Hermitian eigenvalue problem with the SLEPc.EPS.ProblemType.GNHEP object """
        self.eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

        """ Specify the number of eigenvalues to compute
        eps.setDimensions(nev=10, ncv=20, mpd=15)
        mpd (Maximum Projected Dimension): The maximum size of the search subspace. mpd should be less than or equal to ncv.
        ncv (Number of Converged Values): The number of basis vectors in the eigenspace. ncv should be greater than nev.
        """
        self.eps.setDimensions(nev=self.num_eigenvalues)  # Request 10 eigenvalues

        self.eps.setTolerances(tol=self.tol)

        # Set the eigensolver. This is taken from the tutorial: https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_half_loaded_waveguide.html
        # See https://slepc.upv.es/documentation/slepc.pdf for more information on SLEPc
        self.eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

        # Solve the eigenvalue problem
        self.eps.solve()
        self.eps.view()

        # Get number of converged eigenvalues
        nconv = self.eps.getConverged()
        print(f"Number of converged eigenvalues: {nconv}")

        # Get the eigenvalues and eigenvectors
        # print("done")

    def solve_problem(self):
        """Solves the bilinear form."""
        self.solution = fem.Function(self.V)
        a = ufl.lhs(self.LHS)
        if self.RHS != 0:
            L = ufl.rhs(self.RHS)

            fem.solve(a == L, self.u, self.bc)
        else:
            problem = fem.fem.petscLinearProblem(a, self.u, self.bc)
            uh = problem.solve()

        print("solved")

    def save_eigenproblem(self):
        # Save the eigenvalues to a .pkl file
        self.vals = [
            (i, np.sqrt(-self.eps.getEigenvalue(i)))
            for i in range(self.eps.getConverged())
        ]

        # Sort kz by real part
        self.vals.sort(key=lambda x: x[1].real)

        # List to store eigenvalues (real part only for simplicity)
        eigenvalues = [(val.real, val.imag) for _, val in self.vals]

        # Save eigenvalues to a .pkl file
        eigenvalues_filename = os.path.join(
            self.script_dir, "data/eigenvalues_n" + str(self.num_eigenvalues) + ".pkl"
        )
        # Ensure the directory exists
        os.makedirs(os.path.dirname(eigenvalues_filename), exist_ok=True)

        with open(eigenvalues_filename, "wb") as f:
            pickle.dump(eigenvalues, f)

        print(f"Eigenvalues saved to: {eigenvalues_filename}")

    def run(self):
        self.create_mesh_and_function_space()
        print("\n\n Computed mesh and function space \n\n ")
        self.set_constants()
        self.weak_form()
        self.boundary_condition()
        self.solve_eigenvalue_problem()
        self.save_eigenproblem()
        # self.solve_problem()
        # self.save_solution()


class BoundaryFunction:
    @staticmethod
    def boundary(x, on_boundary):
        # Assuming the boundary function is dependent on these parameters
        # The actual function will depend on your specific use case
        return on_boundary


def main():
    eigen_problem = FENicSEigenProblem(
        num_nodes=10,
        domain_type="cube",
        test_mode=False,
    )

    eigen_problem.run()


def test():
    """This is for running the doc test"""
    # Delete the test_run folder if it exists
    output_dir_test = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_run"
    )
    if os.path.exists(output_dir_test):
        shutil.rmtree(output_dir_test)
        time.sleep(5)

    """ Cube """
    eigen_problem = FENicSEigenProblem(
        num_nodes=1000,
        domain_type="cube",
        test_mode=True,
    )

    eigen_problem.run()


def handler(signum, frame):
    raise Exception("Timeout")


if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     if sys.argv[1] == "main":
    #         main()
    #     elif sys.argv[1] == "test":
    #         test()
    #     else:
    #         print(f"Unknown function: {sys.argv[1]}")
    # else:
    #     print("No function specified. You have 90 seconds to input a function.")
    #     signal.signal(signal.SIGALRM, handler)
    #     signal.alarm(90)
    #     try:
    #         user_input = input()
    #         if user_input == "main":
    #             main()
    #         elif user_input == "test":
    #             test()
    #         else:
    #             print(f"Unknown function: {user_input}")
    #     except Exception:
    #         print("No input received in 90 seconds.")
    test()
