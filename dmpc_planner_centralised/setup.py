from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['dmpc_planner_centralised'],
    scripts=['scripts/jetracersimulator_mpc.py'],
    package_dir={'':'scripts'}
)

setup(**d)