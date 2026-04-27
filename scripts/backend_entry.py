#!/usr/bin/env python3
"""Entry point used by PyInstaller to freeze the Flask backend.

PyInstaller command (run from the project root):
    pyinstaller --onefile --name backend_server \\
        --hidden-import flask_cors \\
        --collect-all timm \\
        --add-data "backend;backend" \\
        scripts/backend_entry.py

The resulting dist/backend_server.exe is a self-contained Windows executable
that starts the Flask server on port 5000 without requiring Python installed.

When frozen (sys.frozen == True) PyInstaller extracts the bundle to a temp
directory stored in sys._MEIPASS.  Inserting that directory at the front of
sys.path makes `import backend` work exactly as in normal Python.
"""

import sys
import os

if getattr(sys, "frozen", False):
    # Running as a PyInstaller bundle — add the extraction dir to sys.path
    # so that 'from backend.server import main' resolves correctly.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass and os.path.isdir(meipass):
        sys.path.insert(0, meipass)  # type: ignore[arg-type]

from backend.server import main  # noqa: E402

if __name__ == "__main__":
    main()
