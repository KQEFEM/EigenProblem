"""
This is for solving mask deformation problems using FEniCS (FEM).
"""

import datetime
import os
import shutil
import time
from logging import warn

# from dolfinx.fem import FunctionSpace, dirichletbc, Constant, locate_dofs_topological, TrialFunction, TestFunction# import mshr  # package in fenics
import numpy as np
import ufl as ufl
from dolfinx import fem

#
from dolfinx.mesh import CellType, create_box
from mpi4py import MPI
from slepc4py import SLEPc

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

    def set_experiment_parameters(self):
        """Sets up the experiment"""
        if self.test_problem:
            self.experiment_name = "Test_problem".lower()
            self.subfolder_name = "test_problem"
            self.set_test_parameters()
            self.RHS_test_problem()
            warn("Test problem selected. No parameters set.")
        return self.experiment_name, self.subfolder_name

    def set_test_parameters(self):
        """Simple exact example for u(x,y) = sin(x)sin(y)"""
        self.domain = [2 * np.pi, 2 * np.pi]  # unit square
        self.area = self.domain[0] * self.domain[1]
        self.domain_type = "cube"  # forces unit square

    def set_aluminum_parameters(self):
        """Billy's experiment during his phd"""
        self.domain = [1, 1]
        self.area = self.domain[0] * self.domain[1]
        self.pressures = np.array([101325 * self.area])
        self.thickness = np.array([2.54e-2])
        self.Youngs_Mod = 70e9
        self.Poisson = 0.3
        self.deflection_curve = (
            self.Youngs_Mod * self.thickness**3 / (12 * (1 - self.Poisson**2))
        )

    def set_mm_33_26_parameters(self):
        self.domain = np.array([26e-3, 33e-3])  # set from the origin
        self.area = self.domain[0] * self.domain[1]
        self.thickness = np.array([20e-9])
        self.Youngs_Mod = 166e9
        self.Poisson = 0.23
        self.pressures = self.force_in_newtons / (33e-3 * 26e-3)

    def set_mm_2_2_parameters(self):
        self.domain = np.array([2e-3, 2e-3])
        self.area = self.domain[0] * self.domain[1]
        self.thickness = np.array([20e-9])
        self.Youngs_Mod = 166e9
        self.Poisson = 0.23
        self.pressures = self.force_in_newtons / (33e-3 * 26e-3)

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

    def compute_bilinear_form(self):
        """Computes the bilinear form for the problem."""
        self.RHS = (
            self.rhs_func * self.v * fem.dx
        )  # Note \Delta u = -f in the Poisson equation
        self.LHS = -fem.inner(fem.grad(self.w), fem.grad(self.v)) * fem.dx

    def weak_form(self):
        """Computes the weak form of the problem."""
        self.frequency = fem.Constant(self.mesh, 1.0)  # omega = 0 for unexcited system
        self.permittivity = fem.Constant(self.mesh, 1.0)  # epsilon
        self.permeability = fem.Constant(self.mesh, 1.0)  # mu
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

    def boundary_condition(self):  #
        """Create a Dirichlet boundary condition on the entire boundary. Parameters: V: FunctionSpace - The function space to apply the boundary condition to. value: float - The value of the Dirichlet condition (default is 0.0). Returns: bc: DirichletBC - The boundary condition."""
        # Define the value of the boundary condition
        value = 0.0
        u_D = fem.Constant(self.V.mesh, value)
        # Locate the degrees of freedom on the boundary
        boundary_dofs = fem.locate_dofs_geometrical(
            self.V, lambda x: np.full(x.shape[1], True)
        )
        # Create the boundary condition
        self.bc = fem.dirichletbc(u_D, boundary_dofs, self.V)

    def solve_eigenvalue_problem(self):
        """Solves the eigenvalue problem for the bilinear form using SLEPc."""
        # Create a solution function in the function space
        self.solution = fem.Function(self.V)

        # Define the bilinear form (A) and mass matrix (B), assuming B is identity here
        A = ufl.lhs(self.LHS)  # Stiffness matrix (bilinear form)
        B = ufl.Identity(
            self.V.dofmap().size()
        )  # Mass matrix (identity for standard eigenvalue problems)

        # Mass matrix (identity for standard eigenvalue problems)

        # Convert UFL expressions to matrices (PETSc)
        A_pet = fem.assemble_matrix(A, self.bc)
        B_pet = fem.assemble_matrix(B, self.bc)

        # Convert the matrices to PETSc format
        A_pet.assemble()
        B_pet.assemble()

        # Setup the eigenvalue problem: A u = λ B u
        eig_problem = SLEPc.EPS().create()  # Eigenvalue problem solver (SLEPc)
        eig_problem.setOperators(A_pet, B_pet)  # Set A and B matrices
        eig_problem.setProblemType(
            SLEPc.EPS.ProblemType.GHEP
        )  # Generalized Hermitian Eigenvalue Problem
        eig_problem.setWhichEigenpairs(
            SLEPc.EPS.Which.SMALLEST
        )  # Solve for smallest eigenvalues

        # Solve the eigenvalue problem
        eig_problem.solve()

        # Get the number of converged eigenvalues
        num_eigenvalues = eig_problem.getConverged()

        # Print eigenvalues and eigenvectors (eigenfunctions in FEM)
        for i in range(num_eigenvalues):
            eigenvalue = eig_problem.getEigenvalue(i)
            eigenvector = eig_problem.getEigenvector(i)
            print(f"Eigenvalue {i}: {eigenvalue}")
            print(f"Eigenvector {i}: {eigenvector}")

        print("done")

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

    def save_solution(self):
        """Saves the pvd and vtu files in the appropriate directory."""
        if self.test_mode:
            output_dir_test = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "test_run"
            )
            print(f"The test file is saved {output_dir_test}")
            # Create the directory if it doesn't exist
            os.makedirs(output_dir_test, exist_ok=True)

            # Create a File object for output
            out_file = fem.File(
                os.path.join(
                    output_dir_test, f"{self.experiment_name.lower()}_results.pvd"
                )
            )

            # Save the solution
            out_file << self.w
        else:
            now = datetime.datetime.now()
            date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
            # Create the subfolder if it doesn't exist
            self.subfolder = f"{self.experiment_type}_thick_{str(self.thickness[0]).replace('.', '_')}/fracture_{str(self.prestress).replace('.', '_')}/{self.domain_type}"  # Create the subfolder if it doesn't exist            os.makedirs(os.path.join(self.script_dir, self.subfolder), exist_ok=True)
            fem.File(
                f"{self.subfolder}/membrane_displacement_{self.best_worst_bool}.pvd"
            )

            if self.domain_type == "circle":
                filename2 = os.path.join(
                    self.script_dir,
                    self.subfolder,
                    f"{date_string}_BW_{self.best_worst_bool}_radius_{self.domain[0]}_{self.num_nodes}_solution_displacement.pvd",
                )
            else:
                filename2 = os.path.join(
                    self.script_dir,
                    self.subfolder,
                    f"{date_string}_BW_{self.best_worst_bool}_domain_{self.domain[0]}_{self.domain[1]}_{self.num_nodes}_solution_displacement.pvd",
                )
            vtkfile2 = fem.File(filename2)
            vtkfile2 << self.w

            print(f"Solution displacement saved to: {filename2}")
            print("Solution displacement vector:", self.w.vector().get_local())

    def run(self):
        self.create_mesh_and_function_space()
        print("\n\n Computed mesh and function space \n\n ")
        self.weak_form()
        self.solve_eigenvalue_problem()
        self.solve_problem()
        self.save_solution()


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
