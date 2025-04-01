"""
This module patches the sqlite3 module to use pysqlite3 instead.
This is needed because ChromaDB requires SQLite 3.35.0 or newer,
but some environments (like Streamlit Cloud) have older versions.
"""

import sys
import importlib.util

# Check if pysqlite3 is available
if importlib.util.find_spec("pysqlite3") is not None:
    # If pysqlite3 is available, use it instead of sqlite3
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
    # Print a message to confirm the patch was applied
    print("SQLite has been patched to use pysqlite3")
else:
    print("pysqlite3 not found, using system sqlite3") 