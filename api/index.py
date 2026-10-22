"""
Vercel Serverless Entry Point
Wraps the FastAPI application for deployment on Vercel's Python runtime.
Also works locally via: uvicorn api.index:app
"""

import sys
import os

# Ensure the project root is in the Python path so src.* imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.main import app

# Vercel expects the app object at module level
# This file is the entry point referenced in vercel.json
