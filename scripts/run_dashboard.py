#!/usr/bin/env python
"""Script to run the Streamlit dashboard."""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    # Get the dashboard path
    project_root = Path(__file__).parent.parent
    dashboard_path = project_root / "dashboard" / "app.py"

    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        return 1

    # Run Streamlit
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    print(f"Starting dashboard at {dashboard_path}")
    print("Open http://localhost:8501 in your browser")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
