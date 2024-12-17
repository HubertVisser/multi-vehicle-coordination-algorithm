from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['mpc_planner_py'],
    scripts=['scripts/jackalsimulator_mpc.py'],
    package_dir={'': 'scripts'}
)

setup(**d)