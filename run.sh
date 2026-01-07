#!/bin/bash
set -e

# Run Streamlit from the protected source directory.
# The current working directory (/app) will host user-generated files.
exec streamlit run /usr/local/src/app/app.py --server.port=8501 --server.address=0.0.0.0 --server.enableCORS=false
