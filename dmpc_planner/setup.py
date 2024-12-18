from setuptools import setup

package_name = 'dmpc_planner'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],  # Dependencies are defined here
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='DMPC Planner package for Jetracer with MPC integration in ROS 2.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'jetracer_mpc = dmpc_planner.jackalsimulator_mpc:main',
        ],
    },
    package_dir={'': 'src'},  # Change the root directory for Python source
)