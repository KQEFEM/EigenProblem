import importlib.util

if importlib.util.find_spec("petsc4py") is not None:
    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
    from petsc4py.PETSc import ScalarType  # type: ignore
else:
    print("This demo requires petsc4py.")
    exit(0)

import dolfinx
from mpi4py import MPI
from dolfinx import fem, io, mesh
from ufl import dx, grad, inner
from dolfinx.io import XDMFFile
  
mesh, cell_tag, facet_tag = dolfinx.io.gmshio.read_from_msh('/home/kieran/git/EigenProblem/sphere_mesh.msh', MPI.COMM_WORLD, 0, gdim=3)

# Function to read mesh from an XDMF file
def read_xdmf_mesh(file_path, comm=MPI.COMM_WORLD):
    try:
        with io.XDMFFile(comm, file_path, "r") as xdmf:
            msh = xdmf.read_mesh(name="Grid")
            msh.name = "Imported mesh"
        return msh
    except Exception as e:
        print(f"Error reading mesh from {file_path}: {e}")
        exit(1)

# File path to the XDMF mesh
file_path = "/home/kieran/git/EigenProblem/out_gmsh/mesh.xdmf"

# Read the mesh
msh1 = read_xdmf_mesh(file_path, comm=MPI.COMM_WORLD)

# Check the mesh
print(msh1)
print(type(msh1))

# Create a function space for curl-conforming elements
V = fem.FunctionSpace(msh1, "N1curl", cppV=1)
u = fem.TrialFunction(V)
v = fem.TestFunction(V)

# Define a bilinear form for demonstration
a = inner(grad(u), grad(v)) * dx
print(a)
