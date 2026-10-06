"""Run the recommended three-object simulation: python run.py."""
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parent
    local_python=root/".venv"/("Scripts/python.exe" if sys.platform=="win32" else "bin/python")
    interpreter=str(local_python) if local_python.is_file() else sys.executable
    output=root/"outputs"/f"simulation_{datetime.now():%Y%m%d_%H%M%S_%f}"
    command=[interpreter,"-m","robotic_arm","persistent-pick-place",
             "--model",str(root/"models/panda_simulation_policy.zip"),"--output",str(output),*sys.argv[1:]]
    try:
        return subprocess.call(command,cwd=root)
    except KeyboardInterrupt:
        return 130


if __name__=="__main__":
    raise SystemExit(main())
