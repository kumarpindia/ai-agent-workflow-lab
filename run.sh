#!/bin/bash
cd /home/puneetubuntu24/code/python/qa_agent
source venv/bin/activate
streamlit run app.py --server.port 8501 --server.address 0.0.0.0