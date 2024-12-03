import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx import io, fem, mesh


class MeshingCodes:
    def __init__(self):
        # Physical constants
        self.epsilon_0 = 8.8541878128 * 10**-12  # Vacuum permittivity
        self.mu_0 = 4 * np.pi * 10**-7  # Vacuum permeability
        self.Z0 = np.sqrt(self.mu_0 / self.epsilon_0)  # Vacuum impedance

        # Geometrical constants
        self.radius_sph = 0.025  # Radius of the sphere
        self.radius_dom = 1.0  # Radius of the domain
        self.radius_scatt = 0.4 * self.radius_dom  # Radius for scattering calculation
        self.radius_pml = 0.25  # Radius of the PML shell

        # Mesh sizes
        self.mesh_factor = 1.0
        self.in_sph_size = self.mesh_factor * 2.0e-3
        self.on_sph_size = self.mesh_factor * 2.0e-3
        self.scatt_size = self.mesh_factor * 60.0e-3
        self.pml_size = self.mesh_factor * 40.0e-3

        # Tags for subdomains
        self.au_tag = 1
        self.bkg_tag = 2
        self.pml_tag = 3
        self.scatt_tag = 4

    def generate_mesh_sphere_axis(self):
        """Generate a mesh for a spherical domain with axis symmetry."""
        gmsh.model.add("geometry")

        # Add points
        points = [
            gmsh.model.occ.addPoint(0, self.radius_sph * 0.5, 0, tag=10),
            gmsh.model.occ.addPoint(0, self.radius_sph, 0, tag=8),
            gmsh.model.occ.addPoint(0, self.radius_scatt, 0, tag=6),
            gmsh.model.occ.addPoint(0, self.radius_dom, 0, tag=4),
            gmsh.model.occ.addPoint(0, self.radius_dom + self.radius_pml, 0, tag=2),
            gmsh.model.occ.addPoint(0, -self.radius_sph * 0.5, 0, tag=9),
            gmsh.model.occ.addPoint(0, -self.radius_sph, 0, tag=7),
            gmsh.model.occ.addPoint(0, -self.radius_scatt, 0, tag=5),
            gmsh.model.occ.addPoint(0, -self.radius_dom, 0, tag=3),
            gmsh.model.occ.addPoint(0, -(self.radius_dom + self.radius_pml), 0, tag=1),
        ]

        # Add lines between points
        lines = [
            gmsh.model.occ.addLine(points[i], points[i + 1]) for i in range(len(points) - 1)
        ]
        lines.append(gmsh.model.occ.addLine(points[-1], points[0]))  # Close the loop

        # Synchronize and validate geometry
        gmsh.model.occ.synchronize()
        # gmsh.model.occ.check()

        # Define curve loops and surfaces
        gmsh.model.occ.addCurveLoop(lines, tag=1)
        gmsh.model.occ.addPlaneSurface([1], tag=1)

        # Add physical groups
        gmsh.model.addPhysicalGroup(2, [1], tag=self.au_tag)

        # Synchronize
        gmsh.model.occ.synchronize()

        # Set mesh size
        for p in points:
            gmsh.model.mesh.setSize([(0, p)], self.in_sph_size)

        # Generate mesh
        try:
            gmsh.model.mesh.generate(2)
        except Exception as e:
            print(f"Mesh generation failed: {e}")
            gmsh.write("debug_failure.msh")
            raise

        # Debug visualization
        gmsh.fltk.run()

        return gmsh.model



    def main():
        gmsh.initialize()
        mesher = MeshingCodes()

        model = None
        if MPI.COMM_WORLD.rank == 0:
            # Generate the mesh on the master process
            model = mesher.generate_mesh_sphere_axis()

        # Broadcast the model to all processes
        model = MPI.COMM_WORLD.bcast(model, root=0)

        # Convert GMSH model to a DOLFINx mesh
        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
        msh, cell_tags, facet_tags = io.gmshio.model_to_mesh(
            model, MPI.COMM_WORLD, 0, gdim=2, partitioner=partitioner
        )

        gmsh.finalize()
        MPI.COMM_WORLD.barrier()
        return msh, cell_tags,facet_tags

# Main Execution
if __name__ == "__main__":
    gmsh.initialize()
    mesher = MeshingCodes()

    model = None
    if MPI.COMM_WORLD.rank == 0:
        # Generate the mesh on the master process
        model = mesher.generate_mesh_sphere_axis()

    # Broadcast the model to all processes
    model = MPI.COMM_WORLD.bcast(model, root=0)

    # Convert GMSH model to a DOLFINx mesh
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    msh, cell_tags, facet_tags = io.gmshio.model_to_mesh(
        model, MPI.COMM_WORLD, 0, gdim=2, partitioner=partitioner
    )

    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
