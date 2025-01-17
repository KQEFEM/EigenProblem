"""
This is for solving mask deformation problems using FEniCS (FEM).
"""

import os

# from dolfinx.fem import FunctionSpace, dirichletbc, Constant, locate_dofs_topological, TrialFunction, TestFunction# import self.meshr  # package in fenics
import pickle
import shutil
import time

import basix.ufl as ufl_basis
import dolfinx.fem.petsc as fem_petsc
import numpy as np
import ufl as ufl
from dolfinx import fem
from dolfinx.mesh import CellType, create_box, create_rectangle, exterior_facet_indices, locate_entities
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
    r"""
    This solves for the eigenvalues of the Maxwell problem:
    $\nabla \times \nabla \times \mathbf{E} = \omega^2 \varepsilon \mu \mathbf{E}$
    for the BC of $\nabla \times \mathbf{E} = 0$.

    Running the code on macOS:
        Docker command to start FEniCS:
        docker run -ti -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:current

    The solver uses SLEPc and is based on the tutorial:
    https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_half_loaded_waveguide.html
    See https://slepc.upv.es/documentation/slepc.pdf for more information on SLEPc.

    #? The test mode is not set up yet.
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
          domain_type (str): Type of domain (e.g., 'cube', 'rectangle'). In 3D [height width depth]
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
        self.nodes = None
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
        self.parent_dir = os.path.dirname(self.script_dir)  # Get the parent directory
        self.workspace = os.path.join(self.parent_dir, "EigenProblem")

    def set_constants(self):
        """
        Set the physical constants for the problem.

        This method initializes the frequency, permittivity, and permeability
        constants used in the Maxwell eigenvalue problem.

        Parameters:
        None

        Returns:
        None
        """
        # Frequency of the system (omega)
        height = self.domain[0]
        self.lambda0 = height / 0.2
        self.k0 = fem.Constant(self.mesh, 2 * np.pi / self.lambda0)

        # Electric permittivity (epsilon). This should be an unknown t o solve for
        self.permittivity = fem.Constant(self.mesh, 1.0)

        # Magnetic permeability (mu)
        self.permeability = fem.Constant(self.mesh, 1.0)

    def create_mesh_and_function_space(self):
        """Creates a mesh for the problem.

        The mesh type is selected based on the value of the `domain_type` parameter. The
        mesh is either a rectangle or a cube, and the number of nodes is specified by the
        `num_nodes` parameter.

        Parameters:
        None

        Returns:
        None
        """
        if self.domain_type.lower() == "rectangle":
            # Create a rectangle mesh
            self.mesh = create_rectangle(
                MPI.COMM_WORLD,
                [np.array([0, 0]), np.array([self.domain[0], self.domain[1]])],
                [
                    self.num_nodes,
                    int(self.num_nodes / 300 * 120),
                ],  # Number of elements in each direction
                CellType.quadrilateral,
            )
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim - 1, self.mesh.topology.dim
            )

        elif self.domain_type.lower() == "cube":
            # Create a cube mesh

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
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim - 1, self.mesh.topology.dim
            )
        else:
            print("\nStandard unit square is used.\n")
            # Create a unit square mesh
            self.mesh = fem.UnitSquareMesh(self.num_nodes, self.num_nodes)

        # Create the function space
        self.nodes = self.mesh.geometry.x
        
        D = fem.functionspace(self.mesh, ("DQ", 0))
        self.eps = fem.Function(D)
        eps_v = 1
        eps_d = 2.45
        d = 0.5 * self.domain[0]
        def Omega_d(x):
            return x[1] <= d


        def Omega_v(x):
            return x[1] >= d


        cells_v = locate_entities(self.mesh, self.mesh.topology.dim, Omega_v)
        cells_d = locate_entities(self.mesh, self.mesh.topology.dim, Omega_d)

        self.eps.x.array[cells_d] = np.full_like(cells_d, eps_d )
        self.eps.x.array[cells_v] = np.full_like(cells_v, eps_v)
        # self.V = fem.functionspace(self.mesh, ("P", 1))
        degree = 1
        RTCE = ufl_basis.element("RTCE", self.mesh.basix_cell(), degree)
        Q = ufl_basis.element("Lagrange", self.mesh.basix_cell(), degree)
        self.V = fem.functionspace(self.mesh, ufl_basis.mixed_element([RTCE, Q]))
        # self.V = fem.functionspace(self.mesh, ("Nedelec 1st kind H(curl)", 1))

        self.et, self.ez = ufl.TrialFunctions(self.V)
        self.vt, self.vz = ufl.TestFunctions(self.V)
        # self.u = ufl.TrialFunction(self.V)
        # self.v = ufl.TestFunction(self.V)

    def weak_form(self):
        r"""Computes the weak form of the Maxwell eigenvalue problem.

        This method defines the weak form by calculating the curl-curl
        and mass bilinear forms, then combines them to form the left-hand
        side (LHS) of the equation. The right-hand side (RHS) is set to zero.

        \nabla \times \nabla \times E - \omega^2 \varepsilon \mu E = 0
        """

        a_tt = (
            ufl.inner(ufl.curl(self.et), ufl.curl(self.vt))
            - (self.k0**2) * self.eps * ufl.inner(self.et, self.vt)
        ) * ufl.dx
        b_tt = ufl.inner(self.et, self.vt) * ufl.dx
        b_tz = ufl.inner(self.et, ufl.grad(self.vz)) * ufl.dx
        b_zt = ufl.inner(ufl.grad(self.ez), self.vt) * ufl.dx
        b_zz = (
            ufl.inner(ufl.grad(self.ez), ufl.grad(self.vz))
            - (self.k0**2) * self.eps * ufl.inner(self.ez, self.vz)
        ) * ufl.dx

        self.a = fem.form(a_tt)
        self.b = fem.form(b_tt + b_tz + b_zt + b_zz)

    def set_boundary_condition(self):
        """Create a Dirichlet boundary condition on the entire boundary.

        Parameters
        ----------
        value : float
            The value of the Dirichlet condition (default is 0.0).

        Returns
        -------
        boundary_condition : DirichletBC
            The boundary condition.
        """
        bc_facets = exterior_facet_indices(self.mesh.topology)
        bc_dofs = fem.locate_dofs_topological(
            self.V, self.mesh.topology.dim - 1, bc_facets
        )
        u_bc = fem.Function(self.V)
        with u_bc.x.petsc_vec.localForm() as loc:
            loc.set(0)
        self.bc = fem.dirichletbc(u_bc, bc_dofs)

    def solve_eigenvalue_problem(self):
        """
        Solves the eigenvalue problem for the bilinear form using SLEPc.
        eps.setDimensions(nev=10, ncv=20, mpd=15)
        Specify the number of eigenvalues to compute
        mpd (Maximum Projected Dimension): The maximum size of the search subspace. mpd should be less than or equal to ncv.
        ncv (Number of Converged Values): The number of basis vectors in the eigenspace. ncv should be greater than nev.
        """
        # Create a solution function in the function space
        A = fem_petsc.assemble_matrix(self.a, bcs=[self.bc])
        A.assemble()
        # A.shift(1e-5)
        B = fem_petsc.assemble_matrix(self.b, bcs=[self.bc])
        B.assemble()
        """ 
        If the matrices in the problem have known properties (e.g. hermiticity) we can use this information in SLEPc to accelerate the calculation with the setProblemType function. For this problem, there is no property that can be exploited, and therefore we define it as a generalized non-Hermitian eigenvalue problem with the SLEPc.EPS.ProblemType.GNHEP object 
        """
        eps = SLEPc.EPS().create(self.mesh.comm)
        eps.setOperators(A, B)

        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setTolerances(tol=self.tol)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        # Get ST context from eps
        st = eps.getST()

        # Set shift-and-invert transformation
        st.setType(SLEPc.ST.Type.SINVERT)
        """
                Then, we need to define the number of eigenvalues we want to
        calculate. We can do this with the `setDimensions` function, where we
        specify that we are looking for just one eigenvalue:

        """
        eps.setDimensions(nev=self.num_eigenvalues)
        """

        We can finally solve the problem with the `solve` function. To gain a
        deeper insight over the simulation, we also print an output message
        from SLEPc by calling the `view` and `errorView` function:

        """
        eps.solve()
        eps.view()
        eps.errorView()

        # Get number of converged eigenvalues
        nconv = eps.getConverged()
        print(f"Number of converged eigenvalues: {nconv}")
        self.eps = eps

    def save_eigenproblem(self):
        """
        Saves the eigenvalues to a .pkl file. The eigenvalues are sorted by
        their real part.

        The file is saved in the 'data' folder of the script directory with
        the name 'eigenvalues_nX.pkl', where X is the number of eigenvalues
        computed.
        """
        # Save the eigenvalues to a .pkl file
        self.vals = [
            (i, np.sqrt(-self.eps.getEigenvalue(i)))
            for i in range(self.eps.getConverged())
        ]
        self.vals = [
            (
                i,
                np.sqrt(
                    self.eps.getEigenvalue(i)
                    / float(self.permittivity)
                    / float(self.permeability)
                ),
            )  # Correct
            for i in range(self.eps.getConverged())
        ]

        # Sort kz by real part
        self.vals.sort(key=lambda x: x[1].real)

        # List to store eigenvalues (real part only for simplicity)
        eigenvalues = [(val.real, val.imag) for _, val in self.vals]

        # Save eigenvalues to a .pkl file
        eigenvalues_filename = os.path.join(
            self.parent_dir, "data/eigenvalues_n" + str(self.num_eigenvalues) + ".pkl"
        )
        # Ensure the directory exists
        os.makedirs(os.path.dirname(eigenvalues_filename), exist_ok=True)

        with open(eigenvalues_filename, "wb") as f:
            pickle.dump(eigenvalues, f)

        print(f"Eigenvalues saved to: {eigenvalues_filename}")

    def analytical_eigenfrequencies_rectangle_tm(self, num_modes=5):
        """
        Computes the analytical eigenfrequencies for the 2D TM mode in a rectangle.

        Args:
            num_modes (int): Number of modes to calculate in each direction (default: 5).

        Returns:
            numpy.ndarray: A 2D array containing the eigenfrequencies.
        """
        Lx = self.domain[0]
        Ly = self.domain[1]
        mu = float(self.permeability)  # convert from fenics constant
        epsilon = float(self.permittivity)  # convert from fenics constant

        eigenfrequencies = np.zeros((num_modes, num_modes))

        for m in range(num_modes):
            for n in range(num_modes):
                if m == 0 and n == 0:
                    continue
                omega = (np.pi / np.sqrt(mu * epsilon)) * np.sqrt(
                    (m / Lx) ** 2 + (n / Ly) ** 2
                )
                eigenfrequencies[m, n] = omega

        return eigenfrequencies

    def run(self):
        self.create_mesh_and_function_space()
        print("\n\n Computed mesh and function space \n\n ")
        self.set_constants()
        self.weak_form()
        self.set_boundary_condition()
        self.solve_eigenvalue_problem()
        print('solved')
        # self.analytical_eigenfrequencies_rectangle_tm()
        self.save_eigenproblem()


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
    """
    #! This is for running the doc test and does not work yet
    """
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
    # test() #? THis is not set up to do any tests yet but only runs the main function

    eigen_problem = FENicSEigenProblem(
        num_nodes=300, domain_type="rectangle", test_mode=False, num_eigenvalues=10
    )
    eigen_problem.tol = 1e-9
    eigen_problem.domain = [1, 1, 1]
    eigen_problem.domain = [1, 0.45, 1]
    eigen_problem.run()
    print(eigen_problem.vals)
