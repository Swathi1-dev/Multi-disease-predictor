import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(PROJECT_ROOT)

import streamlit as st
import requests


st.set_page_config(
    page_title="Multi-disease Predictor", page_icon=":hospital:", layout="centered"
)
st.title("Multi-disease Predictor")
st.write(
    """Use the left sidebar to navigate:
    -Diabetes risk predictor
    Heart disease predictor"""
)
st.info("Make sure that FASTAPI backend is running before using predictions.")
