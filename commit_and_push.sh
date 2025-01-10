# commit_and_push.sh
#!/bin/bash

# Get the container ID of the stopped container
container_id=$(docker ps -a -q --filter "name=dmpc_planner-1")

# Commit the container to the same image
docker commit $container_id hubertvisser/dmpc_planner

# Stop the docker-compose services
docker-compose down