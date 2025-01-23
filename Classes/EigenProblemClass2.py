import os

import basix.ufl as ufl_basis
import dolfinx.fem.petsc as fem_petsc

# from dolfinx.fem import FunctionSpace, dirichletbc, Constant, locate_dofs_topological, TrialFunction, TestFunction# import self.meshr  # package in fenics
import numpy as np
import ufl as ufl
from dolfinx import fem
from dolfinx import mesh as dfl_mesh
from mpi4py import MPI
import pickle
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

try:
    import dolfinx
    from petsc4py import PETSc

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
    if PETSc.IntType == np.int64 and MPI.COMM_WORLD.size > 1:
        print(
            "This solver fails with PETSc and 64-bit integers because of memory errors in MUMPS."
        )
        # Note: when PETSc.IntType == np.int32, superlu_dist is used
        # rather than MUMPS and does not trigger memory failures.
        exit(0)

    real_type = PETSc.RealType
    scalar_type = PETSc.ScalarType

except ModuleNotFoundError:
    print("This demo requires petsc4py.")
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
        num_nodes_1D: int = 100,
        domain_type: str = "cube",
        domain: list = None,
        test_problem: bool = False,
        test_mode: bool = False,
        num_eigenvalues=10,
        target_value_bool = False,
        file_type=None
    ):
        """

        Args:
          num_nodes_1D (int): Number of nodes for mesh generation. note that for a d=3D case it will be n^d (default: 100).
          domain_type (str): Type of domain (e.g., 'cube', 'rectangle'). In 3D [height width depth]
          test_problem (bool): Run a predefined test problem (default: False).
          test_mode (bool): Enable testing mode for pytest (default: False).
          num_eigenvalues (int): Number of eigenvalues to compute (default: 10).
          target_value_bool (bool): if True then you should specify the target in line 283 for manually

        """
        self.num_nodes_1D = num_nodes_1D
        self.domain_type = domain_type
        self.test_problem = test_problem
        self.test_mode = test_mode
        self.num_eigenvalues = num_eigenvalues

        # Set default tolerances and placeholders for FEniCS objects
        self.tol = 1e-9
        self.mesh = None
        self.nodes: int = None
        self.V = None
        self.u = None
        self.v = None
        self.bc = None
        self.RHS = None
        self.target_value_bool = target_value_bool

        # Initialise domain and boundary settings
        if domain is None:
            ValueError("The domain is not specified [length, width, height]")
        self.domain = domain  # [length, width, height]
        # self.boundary_function = BoundaryFunction()
        # self.boundary_def = None

        # Experiment-related metadata
        self.experiment_name = None
        self.subfolder_name = None
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.parent_dir = os.path.dirname(self.script_dir)  # Get the parent directory
        self.workspace = os.path.join(self.parent_dir, "EigenProblem")
        self.file_type=file_type

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
        height = self.domain[1]
        self.lambda0 = height / 0.2
        self.k0 = fem.Constant(self.mesh, 2 * np.pi / self.lambda0)

        # Electric permittivity (epsilon).
        # Note that compared to the example we do not need to compute \varepsilon at each node as it is a constant here
        self.permittivity = fem.Constant(self.mesh, 1.0)

        # Magnetic permeability (mu)
        self.permeability = fem.Constant(self.mesh, 1.0)

    def create_domain(self):
        """
        Creates the domain for the problem
        """
        if self.domain_type.lower() == "rectangle":
            # Create a rectangle mesh
            self.mesh = dfl_mesh.create_rectangle(
                MPI.COMM_WORLD,
                [np.array([0, 0]), np.array([self.domain[0], self.domain[1]])],
                [
                    int(self.num_nodes_1D),
                    int(self.num_nodes_1D * 0.4),
                ],  # Number of elements in each direction
                dfl_mesh.CellType.quadrilateral,
            )
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim - 1, self.mesh.topology.dim
            )

        elif self.domain_type.lower() == "cube":
            # Create a cube mesh
            self.mesh = dfl_mesh.create_box(
                MPI.COMM_WORLD,
                [
                    np.array([0, 0, 0]),
                    np.array([self.domain[0], self.domain[1], self.domain[2]]),
                ],
                [int(self.num_nodes_1D), int(self.num_nodes_1D), int(self.num_nodes_1D)],
                dfl_mesh.CellType.hexahedron,
            )
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim - 1, self.mesh.topology.dim
            )
        else:
            print("\nStandard unit square is used.\n")
            # Create a unit square mesh
            self.mesh = fem.UnitSquareMesh(self.num_nodes_1D, self.num_nodes_1D)

        # Create the function space
        self.nodes = self.mesh.geometry.x

    def create_function_space(self):
        """
        Defines the function space. Here nedelec elements are used for the e_t terms and lagrange elements for the e_z.
        """
        degree = 1
        RTCE = ufl_basis.element("RTCE", self.mesh.basix_cell(), degree, dtype=real_type)

        Q = ufl_basis.element(
            "Lagrange", self.mesh.basix_cell(), degree, dtype=real_type
        )
        self.V = fem.functionspace(self.mesh, ufl_basis.mixed_element([RTCE, Q]))

    def weak_form(self):
        """
        This is taken directly from the example: https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_half_loaded_waveguide.html
        """

        et, ez = ufl.TrialFunctions(self.V)
        vt, vz = ufl.TestFunctions(self.V)

        a_tt = (
            ufl.inner(ufl.curl(et), ufl.curl(vt))
            - (self.k0**2) * self.permittivity * ufl.inner(et, vt)
        ) * ufl.dx
        b_tt = ufl.inner(et, vt) * ufl.dx
        b_tz = ufl.inner(et, ufl.grad(vz)) * ufl.dx
        b_zt = ufl.inner(ufl.grad(ez), vt) * ufl.dx
        b_zz = (
            ufl.inner(ufl.grad(ez), ufl.grad(vz))
            - (self.k0**2) * self.permittivity * ufl.inner(ez, vz)
        ) * ufl.dx

        self.a = fem.form(a_tt)
        self.b = fem.form(b_tt + b_tz + b_zt + b_zz)

    def set_boundary_condition(self):
        """

        Perfect electric conductor conditions on the waveguide
        wall:

        """
        bc_facets = dfl_mesh.exterior_facet_indices(self.mesh.topology)
        bc_dofs = fem.locate_dofs_topological(
            self.V, self.mesh.topology.dim - 1, bc_facets
        )
        u_bc = fem.Function(self.V)
        with u_bc.x.petsc_vec.localForm() as loc:
            loc.set(0)
        self.bc = fem.dirichletbc(u_bc, bc_dofs)

    def construct_matrix(self):
        self.A = fem_petsc.assemble_matrix(self.a, bcs=[self.bc])
        self.A.assemble()
        self.B = fem_petsc.assemble_matrix(self.b, bcs=[self.bc])
        self.B.assemble()
        # Compute the conjugate transpose manually
        # Compute the conjugate transpose of A
        A_conj_transpose = self.A.transpose()
        A_conj_transpose = A_conj_transpose.conjugate()

        # Compute the conjugate transpose of B
        B_conj_transpose = self.B.transpose()
        B_conj_transpose = B_conj_transpose.conjugate()

        # Check if both matrices are Hermitian: A = A^H and B = B^H
        is_A_hermitian = self.A.equal(A_conj_transpose)
        is_B_hermitian = self.B.equal(B_conj_transpose)

        print(f"Is matrix A Hermitian? {is_A_hermitian}")
        print(f"Is matrix B Hermitian? {is_B_hermitian}")

    def solving(self):
        """
        Solve the problem with SLEPc

        Here techniques to speed up the computation can be used but this can be considered if needed.
        """
        eps = SLEPc.EPS().create(self.mesh.comm)
        eps.setOperators(self.A, self.B)
        """If the matrices in the problem have known properties (e.g.
        hermiticity) we can use this information in SLEPc to accelerate the
        calculation with the `setProblemType` function. For this problem,
        there is no property that can be exploited, and therefore we define it
        as a generalized non-Hermitian eigenvalue problem with the
        `SLEPc.EPS.ProblemType.GNHEP` object:"""
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setTolerances(tol=self.tol)  # tolerance for the iterative solver
        eps.setType(
            SLEPc.EPS.Type.KRYLOVSCHUR
        )  # ? https://slepc.upv.es/documentation/slepc.pdf
        
        if self.target_value_bool is True:
            # # Get ST context from eps
            st = eps.getST()
            # # Set shift-and-invert transformation
            st.setType(SLEPc.ST.Type.SINVERT)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.setTarget(-((0.5 * self.k0) ** 2))
        
        
        eps.setDimensions(nev=self.num_eigenvalues)
        """

        We can finally solve the problem with the `solve` function. To gain a
        deeper insight over the simulation, we also print an output message
        from SLEPc by calling the `view` and `errorView` function:

        """
        eps.solve()
        eps.view()
        eps.errorView()

        self.vals = [
            (i, np.sqrt(-eps.getEigenvalue(i))) for i in range(eps.getConverged())
        ]

        # Sort kz by real part
        self.vals.sort(key=lambda x: x[1].real)
        
    import pickle

    def save_eigenproblem(self):
        """
        Saves the eigenvalues to a file. The user can choose between saving as a 
        .txt file or a .pkl file. The eigenvalues are sorted by their real part.

        The file is saved in the 'data' folder of the script directory with
        the name 'eigenvalues_nX.<extension>', where X is the number of 
        eigenvalues computed and <extension> is either 'txt' or 'pkl'.
        """
        
        # List to store eigenvalues (real and imaginary parts)
        eigenvalues = [(val.real, val.imag) for _, val in self.vals]

        if self.file_type==None:
            # Ask the user for the preferred file format
            file_format = input("Enter file format to save eigenvalues ('txt' or 'pkl'): ").strip().lower()
        else:
            file_format = self.file_type.replace(".", "").replace(" ", "")
        filename = os.path.join(self.parent_dir, f"data/eigenvalues_n{self.num_eigenvalues}.{file_format}")
        
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if file_format == "txt":
            # Save as a text file
            with open(filename, "w") as f:
                for real, imag in eigenvalues:
                    f.write(f"{real}, {imag}\n")
            print(f"Eigenvalues saved to: {filename}")

        elif file_format == "pkl":
            # Save as a pickle file
            with open(filename, "wb") as f:
                pickle.dump(eigenvalues, f)
            print(f"Eigenvalues saved to: {filename}")

        else:
            print("Invalid file format. Please choose 'txt' or 'pkl'.")

    def run(self):
        self.create_domain()
        self.create_function_space()
        self.set_constants()
        self.weak_form()
        self.set_boundary_condition()
        self.construct_matrix()
        self.solving()
        self.save_eigenproblem()


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
        num_nodes_1D=300,
        domain_type="rectangle",
        domain=[1, 1, 1],
        test_mode=False,
        num_eigenvalues=10,
        target_value_bool=True,
        file_type="txt"
    )
    eigen_problem.tol = 1e-9
    eigen_problem.domain = [1, 1, 1]
    eigen_problem.domain = [1, 0.45, 1]
    eigen_problem.run()
    print(eigen_problem.vals)
