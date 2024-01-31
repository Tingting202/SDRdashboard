import streamlit as st

st.set_page_config(layout="wide")
options = {
    'Off': 0,
    'On': 1
}

col3, col4 = st.columns([10,5])

with col3:
    st.image('Diagram2.jpg', use_column_width=True)

with col4:
    st.markdown('**HR Readiness**')
    st.markdown('**Hospital supplies, infrastructure**')
    st.markdown('**High Quality Delivery and Care**')

st.markdown("---")

col1, col2 = st.columns([10, 5])
with col1:
    st.image('Diagram1.jpg', use_column_width=True)

with col2:
    st.markdown('**Utilization of CHVs**')
    st.markdown('**Referral**')
    st.markdown('**Transportation**')
