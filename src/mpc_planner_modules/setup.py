from setuptools import setup, find_packages

setup(
    name="mpc_planner_modules",             # Package name
    version="0.1.0",                     # Version number
    packages=find_packages(),            # Automatically find sub-packages
    author="Hubert",
    author_email="your_email@example.com",
    description="A package for solver modules",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourrepo",   # Repository URL (optional)
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)