import os, sys
import pathlib

def get_solver_import_paths():
    path = pathlib.Path(__file__).parent.resolve()
    sys.path.append(os.path.join(path, "..", "solver_generator"))
    sys.path.append(os.path.join(path))

def get_solver(idx, solver_name):
    if skip_solver_generation:
        solver_file = solver_path(f"nmpc_solver_{self._idx}.json")
        self._solver_nmpc = AcadosOcpSolver(json_file=solver_file, build=False, generate=False)
        print_success(f"NMPC {self._idx} solver loaded from file")
    else:
        self._solver_nmpc, _ = generate_NMPC_solver.generate(self._idx)
        print_success(f"NMPC {self._idx} solver generated")


