import numpy as np
import sys
from mpi4py import MPI
import os
from dolfinx.io import XDMFFile, gmshio

try:
    import gmsh  # type: ignore
except ImportError:
    print("This demo requires gmsh to be installed")
    exit(0)

class MeshingCodes:

    def __init__(self):
        # Initialize Gmsh
        gmsh.initialize(sys.argv)
        gmsh.option.setNumber("General.Terminal", 0)

        self.model = gmsh.model()
        # Define the folder and filename
        # Get the absolute path of the current directory (the root directory of your project)
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Define the folder and filename relative to the root directory
        self.folder = os.path.join(root_dir, "EigenProblem", "meshes")       
        self.name = "square"
        self.filename = os.path.join(self.folder, self.name+ "_mesh")
        self.ext_msh = ".msh"
        self.xmdf = ".xdmf"

        # Optionally, check if the folder exists and create it if not
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # self.filename = "sphere_mesh.msh"


    def gmsh_cube(self):
            """Create a Gmsh model of a cube and save it to a file.
            Args:
                save_path: Path to save the GMSH model (e.g., "path/to/folder/cube.msh").
            Returns:
                Gmsh model with a cube mesh added and saved to the specified file.
            """
            self.model.add(self.name)
            self.model.setCurrent(self.name)

            # Define the cube with dimensions (1, 1, 1) centered at (0, 0, 0)
            cube = self.model.occ.addBox(-0.5, -0.5, -0.5, 1, 1, 1, tag=1)

            # Synchronize OpenCascade representation with gmsh model
            self.model.occ.synchronize()

            # Add physical marker for cells. It is important to call this
            # function after OpenCascade synchronization
            self.model.add_physical_group(dim=3, tags=[cube])

            # Generate the mesh
            self.model.mesh.generate(dim=3)

            # Save the model to a specified file
            gmsh.write(self.filename + self.ext_msh)

            print(f"Model saved to %s", self.filename + self.ext_msh)



    def create_mesh(self, comm=MPI.COMM_WORLD, model=None, name=None, mode='w'):

        """Create a DOLFINx from a Gmsh model and output to file.
        Args:
            comm: MPI communicator top create the mesh on.
            model: Gmsh model.
            name: Name (identifier) of the mesh to add.
            filename: XDMF filename.
            mode: XDMF file mode. "w" (write) or "a" (append).
        """
        msh, ct, ft = gmshio.model_to_mesh(model, comm, rank=0)
        msh.name = name
        ct.name = f"{msh.name}_cells"
        ft.name = f"{msh.name}_facets"
        with XDMFFile(msh.comm, self.filename, mode) as file:
            msh.topology.create_connectivity(2, 3)
            file.write_mesh(msh)
            file.write_meshtags(
                ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
            )
            file.write_meshtags(
                ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
            )

    def main(self):
        # Instantiate the class and generate the mesh
        mesher = MeshingCodes()

        # Define sphere radius and mesh size
        radius = 1.0    # Radius of the sphere
        mesh_size = 0.1 # Size of the mesh elements

        # Call the method to generate the mesh
        mesher.gmsh_cube()

        mesher.create_mesh(MPI.COMM_WORLD, self.model,  "w")

if __name__ == "__main__":
    mesh = MeshingCodes()
    print(mesh.folder)
    mesh.main()