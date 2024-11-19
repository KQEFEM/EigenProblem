# Specify the path to your Conda environment and requirements file
$condaEnvPath = "C:\Users\kiera\.conda\envs\EigenProblem"  # Replace with the correct path
$requirementsFile = "requirements.txt"

# Activate the Conda environment
Write-Host "Activating Conda Environment..."
& conda activate "$condaEnvPath"

# Install requirements from the requirements.txt file
Write-Host "Installing Requirements..."
& pip install -r "$requirementsFile"