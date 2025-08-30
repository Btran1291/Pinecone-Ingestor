#!/bin/bash
# Wrapper script to run the Streamlit app with proper signal handling and config

export STREAMLIT_BROWSER_SERVER_ADDRESS="localhost"

# Use exec to replace the shell process with the Python process
# This ensures signals (e.g., SIGTERM) are sent directly to Streamlit
exec poetry run streamlit run app.py --server.port=8501 --server.address=0.0.0.0
