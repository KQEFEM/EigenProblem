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

#
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
# assert (
#     np.dtype(PETSc.ScalarType).kind == "c"
# ), "PETSc is not configured for complex numbers."


class FENicSEigenProblem:
    """
    Solving the Membrane equation with BC to be considered:

    \nabla w = -p/(\sigma_0 h)
    w = displacement
    sigma_0 = prestressc
    h + thickness

    Then the billinear form becomes:
    a(w, v) = -\int_{\Omega} \nabla w \cdot \nabla v dx
    L(v) = -\int_{\Omega} p/(\sigma_0 h) v dx

     Parameters:
        num_nodes (int): The number of nodes in the mask. Default is 100.
        experiment_type (str): The type of experiment to be conducted.
        domain_type (str): The type of domain.
        test_problem (bool): Whether to run the test problem when the code has been changed. If selected then all values are set for you. Default is False.
        test_mode (bool): Only for testing purposes in pytest. Default is False.

    Running the code on MACOS:
        Docker command to start fenics
        docker run -ti -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:current


    To test the function on a known solution, you can compute the relative error in the following:
    # Create a triangulation object
    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1])

    # Create a pseudocolor plot of the solution
    plt.figure(figsize=(10, 10))
    plt.tripcolor(triangulation, sol)
    plt.colorbar(label='Solution')
    plt.title('2D Solution')
    plt.show()

    exact_solution_func = lambda x, y: np.sin(x) * np.sin(y)
    exact_solution = exact_solution_func(
        nodes[:, 0], nodes[:, 1]
    )

    np.linalg.norm(sol - exact_solution) / np.linalg.norm(exact_solution)
    """

    def __init__(
        self,
        num_nodes: int = 100,
        domain_type: str = "cube",
        test_problem: bool = False,
        test_mode: bool = False,
        num_eigenvalues = 10,
    ):
        self.test_mode = test_mode  # Only for testing purposes in pytest
        self.mesh = None
        self.domain_type = domain_type
        self.domain = []
        self.experiment_name = None
        self.subfolder_name = None
        self.num_nodes = num_nodes
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.V = None  # Function space
        self.u = None  # Trial function
        self.v = None  # Test function
        self.bc = None  # B oundary condition #! Only Dirichlet implemented
        self.RHS = None  # Right-hand side
        self.boundary_function = (
            BoundaryFunction()
        )  # ? Needs extending and understanding
        self.boundary_def = None
        self.test_problem = test_problem
        self.tol = 1e-9
        self.num_eigenvalues = num_eigenvalues

    # def set_experiment_parameters(self):
    #     """Sets up the experiment"""
    #     if self.test_problem:
    #         self.experiment_name = "Test_problem".lower()
    #         self.subfolder_name = "test_problem"
    #         self.set_test_parameters()
    #         self.RHS_test_problem()
    #         warn("Test problem selected. No parameters set.")
    #     return self.experiment_name, self.subfolder_name

    # def set_test_parameters(self):
    #     """Simple exact example for u(x,y) = sin(x)sin(y)"""
    #     self.domain = [2 * np.pi, 2 * np.pi]  # unit square
    #     self.area = self.domain[0] * self.domain[1]
    #     self.domain_type = "cube"  # forces unit square

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
        eigenvalues_filename = os.path.join(self.script_dir, "data/eigenvalues_n" + str(self.num_eigenvalues) +".pkl")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(eigenvalues_filename), exist_ok=True)

        with open(eigenvalues_filename, "wb") as f:
            pickle.dump(eigenvalues, f)

        print(f"Eigenvalues saved to: {eigenvalues_filename}")
        # List to store kz values
        # kz_list = []

        # for i, kz in self.vals:
        #     # Save eigenvector in eh
        #     self.eps.getEigenpair(i, eh.vector)

        #     # Compute error for i-th eigenvalue
        #     error = eps.computeError(i, PETSc.EPS.ErrorType.RELATIVE)

        #     # Verify and save solution
        #     if error < tol and np.isclose(kz.imag, 0, atol=tol):
        #         kz_list.append(kz)

        #         # # Verify if kz is consistent with the analytical equations
        #         # assert verify_mode(kz, w, h, d, lmbd0, eps_d, eps_v, threshold=1e-4)

        #         print(f"eigenvalue: {-kz**2}")
        #         print(f"kz: {kz}")
        #         print(f"kz/k0: {kz / k0}")

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
    membrane_deformation = FENicSEigenProblem(
        num_nodes=10,
        domain_type="cube",
        test_mode=False,
    )

    membrane_deformation.run()


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
    membrane_deformation = FENicSEigenProblem(
        num_nodes=1000,
        domain_type="cube",
        test_mode=True,
    )

    membrane_deformation.run()


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
