# Specify the environment name and requirements file
$envName = "EigenProblem"
$requirementsFile = "requirements.txt"

# Check if the environment exists
if (!(Get-CimInstance -ClassName Win32_Environment -Filter "Name='$envName'")) {
    # Create the Conda environment
    Write-Host "Creating Conda Environment..."
    & mamba create -n "$envName" python=3.10  # Adjust Python version as needed
} else {
    Write-Host "Environment '$envName' already exists."
}

# Activate the Conda environment
Write-Host "Activating Conda Environment..."
& mamba activate "$envName"

# Install requirements from the requirements.txt file
Write-Host "Installing Requirements..."
& mamba env update -f "$requirementsFile" -n "$envName"
