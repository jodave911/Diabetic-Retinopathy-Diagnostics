import streamlit as st

from st_pages import Page, show_pages, add_page_title

add_page_title("Home")

# Specify what pages should be shown in the sidebar, and what their titles 
# and icons should be

st.set_page_config(
    page_title="Diabetic Retinopathy",
    page_icon="👋",
)

st.write("# Diabetic Retinopathy Diagnosis 👁️")

st.markdown(
    """
    Our aim is to develop a hybrid image enhancement algorithm using deep learning for accurate diabetic retinopathy diagnosis from fundus images.
   
    - Check eye fundus for DR or NoDR [Diagnosis](https://diabetic-retinopathy-diagnostics-911.streamlit.app/Diagnostics)
    
   
"""
)
