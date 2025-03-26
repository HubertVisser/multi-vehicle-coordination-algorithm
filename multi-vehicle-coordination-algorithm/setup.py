from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['multi-vehicle-coordination-algorithm'],
    scripts=['scripts/centralised-algorithm/main_centralised.py','scripts/distributed-algorithm/main_distributed.py'],
    package_dir={'':'scripts'}
)

setup(**d)