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

from mpi4py import MPI

# +
import numpy as np

import ufl
import MeshingCodes as meshing
from dolfinx import fem, io, mesh, plot
from dolfinx.mesh import create_box
from dolfinx.fem.petsc import LinearProblem

from ufl import ds, dx, grad, inner


msh1 = create_box(
    comm=MPI.COMM_WORLD,
    points=[(0.0, 0.0, 0.0), (3.0, 2.0, 1.0)],
    n=[30, 20, 10],
    cell_type=mesh.CellType.tetrahedron,
)

msh1 = meshing.main()

print(msh1)

print(type(msh1))
element = ufl.finiteelement("N1curl", msh1.ufl_cell(), 1)
V = fem.FunctionSpace(msh1, element)# Define basis and bilinear form
u = fem.TrialFunction(V)
v = fem.TestFunction(V)