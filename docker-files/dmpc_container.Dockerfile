# Use the official ROS Noetic base image
FROM ros:noetic-robot

# Create a new user and give sudo permissions
RUN useradd -ms /bin/bash dock_user && \
    usermod -aG sudo dock_user && \
    echo "dock_user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the user to the newly created user
USER root

# Install Python 3.8.10
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-dev python3.8-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --set python3 /usr/bin/python3.8

# Install pip for Python 3.8
RUN apt-get install -y python3-pip && \
    python3.8 -m pip install --upgrade pip

    # Set the working directory

WORKDIR /home/dock_user/

# Create a ROS workspace
RUN mkdir -p /home/dock_user/ros_ws/src && \
    cd /home/dock_user/ros_ws && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"
    
# Clone and install acados
RUN apt install -y git && \
    git clone https://github.com/acados/acados.git && \
    cd acados && \
    git submodule update --recursive --init &&\
    mkdir -p build &&\
    cd build &&\
    cmake -DACADOS_SILENT=ON .. &&\
    make install -j4 &&\
    pip install -e /home/dock_user/acados/interfaces/acados_template

# Install dependencies for downloading and extracting tera_renderer
RUN apt-get update && apt-get install -y wget

# Download and install tera_renderer
RUN wget -q https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux -O /home/dock_user/acados/bin/t_renderer && \
    chmod +x /home/dock_user/acados/bin/t_renderer && \
    
RUN pip install pandas
    pip install colorama
    pip install cvxpy


USER dock_user

# Source the ROS setup script
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Set the entrypoint
# Add environment variables to ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/acados/lib" >> ~/.bashrc && \
    echo "export ACADOS_SOURCE_DIR=$HOME/acados/" >> ~/.bashrc

CMD ["tail", "-f", "/dev/null"]
