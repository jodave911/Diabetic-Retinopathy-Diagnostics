import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Diabetic Retinopathy Diagnosis 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    To develop a hybrid image enhancement algorithm using deep learning for accurate diabetic retinopathy diagnosis from fundus images.
   
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [Diagnosis](https://diabetic-retinopathy-diagnostics-911.streamlit.app/Diagnostics)
    
   
"""
)
