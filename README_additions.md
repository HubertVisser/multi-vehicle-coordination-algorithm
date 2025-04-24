install rospackages
- robot_localization with `sudo apt-get install ros-noetic-robot-localization`
1. create venv `python3 -m venv .venv --system-site-packages`
2. install pkgs in requirements.txt `pip install -r requirements.txt`
3. install the following packages
- solver_generator (local)
- mpc_planner_modules (local)
- rospkg

Additions to readme:
(in ros_ws/src/) clone hubertvisser/multi-vehicle-coordination-algorithm
catkin_make in src
source devel/setup.bash

(in ros_ws/src/) clone dart_simulator_pkg (https://github.com/HubertVisser/DART.git)

DART/DART_dynamic_models/
pip install dist/DART_dynamic_models-0.1.0-py3-none-any.whl

source ROS workspace


