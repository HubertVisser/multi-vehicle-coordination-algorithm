from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['dmpc_planner_decentralised'],
    scripts=['scripts/main.py'],
    package_dir={'':'scripts'}
)

setup(**d)