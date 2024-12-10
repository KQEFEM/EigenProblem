import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, io, nls
import dolfinx.fem.petsc
import dolfinx.nls.petsc
from dolfinx.mesh import create_box, CellType
from ufl import (
    dot,
    curl,
    dx,
    TrialFunction,
    TestFunction,
)

# Mesh parameters
L = 3.0
N = 4
mesh = create_box(
    MPI.COMM_WORLD,
    [[-0.5, -0.5, 0.0], [0.5, 0.5, L]],
    [N, N, 4 * N],
    CellType.hexahedron,
)

dim = mesh.topology.dim
print(f"Mesh topology dimension d={dim}.")

degree = 1
shape = (dim,)
V = fem.functionspace(mesh, ("P", degree, shape)) # Lagrange polynomials of degree 1 in 3D
# Define basis and bilinear form
u = TrialFunction(V)
v = TestFunction(V)

# Define problem parameters (frequency set to 0 for unexcited system)
frequency = fem.Constant(mesh, 1.0)  # omega = 0 for unexcited system
permittivity = fem.Constant(mesh, 1.0)  # epsilon
permeability = fem.Constant(mesh, 1.0)  # mu

# Define a bilinear form for demonstration
# Assume that all the boundaries are \nabla \times E = 0
# Bilinear Form 
# """ 
# Next we define the variationally formulation:

# $$ \nabla \times  \left(\frac{1}{\mu} \nabla \times \mathbf{E}(\mathbf{r},\omega) \right) = \omega^2 \varepsilon \mathbf{E}(\mathbf{r},\omega) $$ 
# $$ \nabla \times E \times n = g_N \quad \partial \Omega_N$$ 
# $$ E \times n = g_D \quad \partial \Omega _D $$

# Through integration by parts we get,
# $$ \int_\Omega (\nabla \times \underline{E}) \cdot \nabla \times \underline{v}\ d\Omega - \int_{\partial\Omega_N}   (\nabla \times \underline{E}) \cdot (\nabla \times \underline{v} \ \partial\Omega_N - \omega^2 \varepsilon \int_\Omega \underline{E}\underline{v}\ d\Omega= 0$$ 
# ,
# by using: https://en.wikipedia.org/wiki/Triple_product
# """

# Define curl-curl and mass bilinear forms
curl_curl = dot(curl(u), curl(v)) * dx
mass = frequency**2 * permittivity * permeability * dot(u, v) * dx

# Assemble matrix
a = curl_curl
b = mass
A = fem.assemble_matrix(a + b)
A.assemble()  # Ensure the matrix is finalized

# Create eigensolver
eigensolver = dolfinx.nls.petsc.SLEPcEigenSolver(A)

# Number of eigenvalues to compute
n = 2  # Compute the two lowest eigenvalues
print(f"Computing the first {n} eigenvalues. This can take a minute.")
eigensolver.solve(n)

# Extract and print eigenvalues and frequencies
for i in range(n):
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    frequency = np.sqrt(r)  # Eigenvalue corresponds to ω², so take sqrt to get ω
    print(f"{i+1}. Eigenvalue: {r}, Frequency: {frequency} Hz")

    # Initialize function and assign eigenvector
    u = fem.Function(V)
    u.vector().setArray(rx)  # Assign the real part of the eigenvector

    # Visualize eigenfunction (simple plot or save to file)
    # Example: Save solution to VTK
    with io.XDMFFile(MPI.COMM_WORLD, f"eigenfunction_{i}.xdmf", "w") as xdmf_file:
        xdmf_file.write_mesh(mesh)
        xdmf_file.write_function(u)
