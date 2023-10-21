import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd
import altair as alt

st.set_page_config(layout="wide", page_title="ABM Version 2 Simulation Results", page_icon="📈")
st.sidebar.header("ABM Version 2 Simulation Results")

#SET UP SLIDERS
st.subheader('Sliders')
with st.form('Test'):
    left, right = st.columns(2)
    with left:
        beta_1 = st.slider('beta_1', 0.0, 1.0, 0.1)
    with right:
        beta_2 = st.slider('beta_2', 0.0, 10.0, 0.1)
    submitted = st.form_submit_button("Run")

    t = np.arange(5)
    k1 = beta_2 + t**1 * beta_1
    k2 = beta_2 + t**2 * beta_1
    k3 = beta_2 + t**3 * beta_1

    df = pd.DataFrame(
        {'t':t,'k1':k1, 'k2':k2, 'k3':k3}
    )

    df_melt = pd.melt(df, id_vars=['t'], value_vars=['k1', 'k2', 'k3'])

    chart = (
        alt.Chart(
            data=df_melt,
            title="Testing",
        )
        .mark_line()
        .encode(
            x=alt.X("t", axis=alt.Axis(title="Time")),
            y=alt.Y("value", axis=alt.Axis(title="K value")),
            color= "variable"
        )
    )

    if submitted:
        st.altair_chart(chart, use_container_width=True)
        #st.line_chart(df_melt, x = "t", y = "value", color = "variable")


