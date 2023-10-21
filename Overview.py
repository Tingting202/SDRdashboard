import streamlit as st
import geopandas as gpd
import plotly.express as px
import pandas as pd
import altair as alt

import time
import matplotlib.pyplot as plt

import numpy as np

if 'transport_step' not in st.session_state:
    st.session_state.transport_step = 0

st.set_page_config(layout="wide")
with st.sidebar:
    st.title('Service Delivery Redesign')

left, middle, right = st.columns((2, 5, 2))
with middle:
    st.write('**Service Delivery Redesign** is a cross-sectoral strategy led by Kakamega County that'
             ' tests how to improve maternal and newborn survival by shifting where and when mothers '
             'access care. Redesign focuses on improving the distribution of financing, staff, '
             'equipment, beds and medicines to ensure that women receive ‘right place, high quality '
             'care’ and mothers deliver in or as close as possible to well-functioning hospitals '
             '(CEmONC Level 4/5).')

