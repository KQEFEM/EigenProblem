#!/bin/bash

# Script to manage the FEniCS Docker container
IMAGE_NAME="fenics"
CONTAINER_NAME="fenicsx"
DOCKERFILE_DIR="./docker" # Directory containing the Dockerfile

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install it before running this script."
    exit 1
fi

# Function to build the Docker image
build_image() {
    echo "Building the Docker image in the current workspace..."
    docker build -t $IMAGE_NAME -f $DOCKERFILE_DIR/dockerfile .
}

# Function to run the container in the current workspace
run_container() {
    echo "Starting the Docker container in the current workspace..."
    docker run -it --name $CONTAINER_NAME -v $(pwd):/workspace -w /workspace $IMAGE_NAME
}

# Function to restart a stopped container
restart_container() {
    echo "Restarting the Docker container..."
    docker start -ai $CONTAINER_NAME
}

# Function to delete the container
delete_container() {
    echo "Deleting the Docker container..."
    docker rm -f $CONTAINER_NAME
}

# Function to delete the image
delete_image() {
    echo "Deleting the Docker image..."
    docker rmi -f $IMAGE_NAME
}

# Function to list containers
list_containers() {
    echo "Listing all Docker containers..."
    docker ps -a
}

# Function to stop and remove all containers
close_all_containers() {
    echo "Stopping and removing all Docker containers..."
    docker stop $(docker ps -aq)
    docker rm $(docker ps -aq)
}

# Function to delete both the container and the image
delete_image_container() {
    echo "Deleting the Docker container and image..."
    
    # Remove the container if it exists
    docker rm -f $CONTAINER_NAME
    
    # Remove the image
    docker rmi -f $IMAGE_NAME
}

# Function to update the Docker image if needed
update_image() {
    echo "Checking for updates to the Docker image..."

    # Try pulling the latest image from the registry
    docker pull $IMAGE_NAME

    # Check if the pull was successful by comparing image IDs
    local_image_id=$(docker images -q $IMAGE_NAME)
    remote_image_id=$(docker inspect --format '{{.Id}}' $IMAGE_NAME)

    if [ "$local_image_id" != "$remote_image_id" ]; then
        echo "Remote image is newer. Rebuilding locally..."
        build_image  # Rebuild the image if the local one is outdated
    else
        echo "Docker image is already up-to-date."
    fi
}


# Help menu
help_menu() {
    echo "Usage: ./run.sh [command]"
    echo "Commands:"
    echo "  build                      - Build the Docker image in the current workspace."
    echo "  build clean                - Perform a full reinstallation with no cache."
    echo "  run                        - Run the Docker container."
    echo "  restart                    - Restart a stopped container."
    echo "  delete_container           - Delete the Docker container."
    echo "  delete_image               - Delete the Docker image."
    echo "  list                       - List all Docker containers."
    echo "  close_all                  - Stop and remove all Docker containers."
    echo "  delete_image_container     - Delete both the container and the image."
    echo "  update                     - Update the Docker image (if needed)."
    echo "  help                       - Display this help menu."
}

# Parse command-line arguments
case $1 in
    build) build_image ;;
    run) run_container ;;
    restart) restart_container ;;
    delete_container) delete_container ;;
    delete_image) delete_image ;;
    list) list_containers ;;
    close_all) close_all_containers ;;
    delete_image_container) delete_image_container ;;
    update) update_image ;;
    help | *) help_menu ;;
esac
