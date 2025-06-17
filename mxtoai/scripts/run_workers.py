#!/usr/bin/env python
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def find_tasks_modules():
    """Find all task modules in the project."""
    base_dir = Path(__file__).parent.parent  # mxtoai directory
    tasks_modules = []

    # Look for any Python files that might contain tasks
    for path in base_dir.rglob("*.py"):
        if path.is_file():
            # Convert path to module notation
            relative_path = path.relative_to(base_dir.parent)
            module_path = str(relative_path).replace("/", ".").replace(".py", "")
            tasks_modules.append(module_path)

    return tasks_modules


if __name__ == "__main__":
    # Get all task modules
    modules = find_tasks_modules()

    for _module in modules:
        pass

    # Construct the dramatiq command
    cmd = [
        "dramatiq",
        "--processes",
        str(os.getenv("DRAMATIQ_PROCESSES", "8")),
        "--threads",
        str(os.getenv("DRAMATIQ_THREADS", "8")),
        "--use-spawn",  # Use spawn instead of fork to avoid subprocess issues
        "--watch",
        str(Path(__file__).parent.parent),  # Watch the mxtoai directory
    ]

    # Add any queues if specified
    queues = os.getenv("DRAMATIQ_QUEUES")
    if queues:
        cmd.extend(["--queues", *queues.split(",")])

    # Run dramatiq with connection retry logic
    delay = 1
    while True:
        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode == 3:  # Connection error
                subprocess.run(["sleep", str(delay)], check=False)
                delay = min(delay * 2, 30)  # Exponential backoff, max 30 seconds
            else:
                sys.exit(result.returncode)
        except KeyboardInterrupt:
            sys.exit(0)
