# This file is kept for backward compatibility
# The application has been refactored into multiple modules for better maintainability

import streamlit as st
import sys
import os

# Display a message about the refactoring
st.title("Memory Chatbot - Refactored")
st.info("""
This application has been refactored into a more production-ready structure.
The main application is now in `app.py`.

The refactored application includes:
- Better error handling and logging
- Asynchronous API calls for improved performance
- Qdrant vector database integration (with FAISS fallback)
- Input validation and sanitization
- Docker support for containerized deployment
- Improved configuration management
""")

# Add a button to run the new app
if st.button("Launch New Application"):
    # Redirect to the new app
    os.system(f"{sys.executable} -m streamlit run app.py")
    st.success("Launching new application...")
    st.stop()

# Or automatically redirect
import time
st.write("Redirecting to new application in 5 seconds...")
time.sleep(5)
os.system(f"{sys.executable} -m streamlit run app.py")
st.stop()
