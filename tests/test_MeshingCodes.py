import unittest
from unittest.mock import patch, MagicMock
import os
import sys

from git.EigenProblem.Classes.MeshingCodes import *

class TestMeshingCodes(unittest.TestCase):

    @patch("gmsh.initialize")  # Mock GMSH initialization
    @patch("gmsh.model")  # Mock gmsh.model()
    def test_gmsh_cube_creation(self, mock_model, mock_initialize):
        # Mocking gmsh's model and the methods that should be called
        mock_gmsh = MagicMock()
        mock_model.return_value = mock_gmsh
        mesher = MeshingCodes()

        # Mocking mesh generation behavior
        mock_gmsh.mesh.generate.return_value = None
        mock_gmsh.write.return_value = None

        # Call the method under test
        mesher.gmsh_cube()

        # Assertions to check if the correct methods were called
        mock_gmsh.occ.addBox.assert_called_once_with(-0.5, -0.5, -0.5, 1, 1, 1, tag=1)
        mock_gmsh.mesh.generate.assert_called_once_with(dim=3)
        mock_gmsh.write.assert_called_once_with(mesher.filename + mesher.ext_msh)

    @patch("gmsh.write")  # Mock GMSH write method
    @patch("dolfinx.io.XDMFFile")  # Mock XDMFFile
    @patch("gmshio.model_to_mesh")  # Mock gmshio.model_to_mesh
    def test_create_mesh(self, mock_gmshio, mock_XDMFFile, mock_gmsh_write):
        # Mocking the GMSH model and the output mesh
        mock_msh = MagicMock()
        mock_ct = MagicMock()
        mock_ft = MagicMock()
        mock_gmshio.return_value = (mock_msh, mock_ct, mock_ft)
        mock_XDMFFile.return_value.__enter__.return_value = MagicMock()

        mesher = MeshingCodes()

        # Mocking mesh save and file write behavior
        mock_XDMFFile.return_value.__enter__.return_value.write_mesh.return_value = None
        mock_XDMFFile.return_value.__enter__.return_value.write_meshtags.return_value = None

        # Call the method under test
        mesher.create_mesh(comm=None, model=None, name="test_mesh", mode="w")

        # Assertions to check if the correct methods were called
        mock_XDMFFile.return_value.__enter__.return_value.write_mesh.assert_called_once()
        mock_XDMFFile.return_value.__enter__.return_value.write_meshtags.assert_called()
        mock_gmshio.assert_called_once()

    @patch("meshing_codes.MeshingCodes.gmsh_cube")  # Mock gmsh_cube to avoid actual mesh generation
    @patch("meshing_codes.MeshingCodes.create_mesh")  # Mock create_mesh to avoid writing files
    def test_main(self, mock_create_mesh, mock_gmsh_cube):
        # Create an instance of the MeshingCodes class
        mesher = MeshingCodes()

        # Mock the main method dependencies
        mock_gmsh_cube.return_value = None
        mock_create_mesh.return_value = None

        # Call the main method
        mesher.main()

        # Assertions to ensure that gmsh_cube and create_mesh were called
        mock_gmsh_cube.assert_called_once()
        mock_create_mesh.assert_called_once()

if __name__ == "__main__":
    unittest.main()
