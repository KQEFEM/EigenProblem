import gmsh
import numpy as np
import sys

class MeshingCodes:
    
    def __init__(self):
        # Initialize Gmsh
        gmsh.initialize(sys.argv)
    
    def generate_mesh_sphere(self, radius: float, mesh_size: float):
        # Create geometry for a sphere
        gmsh.model.add("sphere")
        
        # Define the sphere using a simple Gmsh sphere function
        gmsh.model.occ.addSphere(0, 0, 0, radius, tag=1)  # Sphere at (0,0,0) with given radius
        
        # Synchronize the geometry to prepare for meshing
        gmsh.model.occ.synchronize()
        
        # Define mesh size
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), size=mesh_size)
        
        # Generate the mesh
        gmsh.model.mesh.generate(3)  # Generate a 3D mesh
        
        # Optionally, save the mesh to a file
        gmsh.write("sphere_mesh.msh")
        
        # Finalize Gmsh
        gmsh.finalize()
        
        print("Mesh generated and saved as 'sphere_mesh.msh'")

    def main():
        # Instantiate the class and generate the mesh
        mesher = MeshingCodes()
        
        # Define sphere radius and mesh size
        radius = 1.0    # Radius of the sphere
        mesh_size = 0.1 # Size of the mesh elements
        
        # Call the method to generate the mesh
        return mesher.generate_mesh_sphere(radius=radius, mesh_size=mesh_size)

# if __name__ == "__main__":
#     pass
