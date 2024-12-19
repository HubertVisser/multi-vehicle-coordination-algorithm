import os, sys
import pathlib

def get_solver_import_paths():
    path = pathlib.Path(__file__).parent.resolve()
    sys.path.append(os.path.join(path, "..", "solver_generator"))
    sys.path.append(os.path.join(path))