import streamlit as st
import geopandas as gpd
import plotly.express as px
import pandas as pd
import altair as alt
from scipy import interpolate
import time

import numpy as np
st.set_page_config(layout="wide", page_title="ABM LP Simulation Results", page_icon="📈")

st.sidebar.header("ABM LP Simulation Results")

shapefile_path2 = '/Users/tingtingji/Library/CloudStorage/Dropbox/Phd PolyU/My Projects/Postdoc in JHU/SDR_project/Dashboard/sdr_streamlit 2/ke_subcounty.shp'

data2 = gpd.read_file(shapefile_path2)
data2 = data2.loc[data2['county'] == 'Kakamega',]
data2['color'] = np.random.random(len(data2['county']))
data2['color2'] = np.random.random(len(data2['county']))

fig = px.choropleth_mapbox(data2,
                           geojson=data2.geometry,
                           locations=data2.index,
                           mapbox_style="open-street-map",
                           center={'lon': 34.785511, 'lat': 0.430930},
                           zoom=8)

fig_2 = px.choropleth_mapbox(data2,
                           geojson=data2.geometry,
                           locations=data2.index,
                           mapbox_style="open-street-map",
                           color=data2.color,
                           center={'lon': 34.785511, 'lat': 0.430930},
                           zoom=8)

fig_3 = px.choropleth_mapbox(data2,
                           geojson=data2.geometry,
                           locations=data2.index,
                           mapbox_style="open-street-map",
                           color=data2.color2,
                           center={'lon': 34.785511, 'lat': 0.430930},
                           zoom=8)

col1, col2, col3 = st.columns(3)
with col1:
    st.write('Map of Kakamega by Subcounty')
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

with col2:
    st.write('Map of Kakamega by Subcounty - Shift in Deliveries')
    st.plotly_chart(fig_2, theme = 'streamlit', use_container_width=True)
with col3:
    st.write('Map of Kakamega by Subcounty - Adjusted Shift in Deliveries')
    st.plotly_chart(fig_3, theme='streamlit', use_container_width=True)

mmr_all = pd.read_excel('SDR_Plot.xlsx', sheet_name='all mothers', header=None)
mmr_fac = pd.read_excel('SDR_Plot.xlsx', sheet_name='mothers in facilities', header=None)
mmr_qual = pd.read_excel('SDR_Plot.xlsx', sheet_name='mmr_quality', header=None)
mmr_fanc = pd.read_excel('SDR_Plot.xlsx', sheet_name='anc_facility', header=None)
mmr_anc = pd.read_excel('SDR_Plot.xlsx', sheet_name='anc_all', header=None)

# interpolate maternal mortality rate and quality of care
x = mmr_qual[0]
y = mmr_qual[1]
f = interpolate.interp1d(x, y)

# interpolate maternal mortality rate and facility anc
x = mmr_fanc[0]
y = mmr_fanc[1]
g = interpolate.interp1d(x,y)

# interpolate maternal mortality rate and all anc
x = mmr_anc[0]
y = mmr_anc[1]
h = interpolate.interp1d(x,y)

def calc_effect(anc, sdr):
    """Calculate effect of ANC, SDR on maternal mortality rates"""
    # get number at home
    home = mmr_all - mmr_fac
    # get reduction at home
    r_a_home=(h(anc)-g(anc))/(0.1171 - 0.0421)
    # get reduction at facility
    r_a_fac=g(anc)/0.0421
    # get reduction at facility
    r_s_fac = f(sdr)/0.0406
    # add to get adjusted mmr according to reductions
    new_mmrall = home*r_a_home + mmr_fac*r_a_fac*r_s_fac
    new_mmrfac = mmr_fac*r_a_fac*r_s_fac

    return new_mmrall, new_mmrfac

st.subheader('**Scenario Projection**')
left, right = st.columns(2)
with left:
    with st.form('Test'):
        transport_step = 100 / 19
        transport = st.slider(label="Transport Intervention - Select a value:",
                              min_value=0.0,
                              max_value=100.0,
                              value=0.0,
                              step=transport_step)
        quality_effect = st.slider(label='Quality care Intervention - Select a value:',
                                   min_value=0.0,
                                   max_value=75.0,
                                   step=0.5)
        anc_effect = st.slider(label='ANC Intervention - Select a value:',
                                   min_value=0.0,
                                   max_value=100.0,
                                   step=0.5)
        submitted = st.form_submit_button("Test Intervention")

    new_mmrall, new_mmrfac = calc_effect(anc_effect, quality_effect)

    transport_index = int(transport / transport_step)
    mmr_plot = pd.concat([mmr_all.iloc[:, 0], mmr_fac.iloc[:, 0],
                          new_mmrall.iloc[:, transport_index], new_mmrfac.iloc[:, transport_index]],
                         axis=1)
    mmr_plot.columns = ['MMR All', 'MMR Facility', 'MMR All-Intervention', 'MMR Facility-Intervention']
    mmr_plot['x'] = np.linspace(0,90,4)
    mmr_plot = mmr_plot.melt(id_vars=['x'], value_vars=[
        'MMR All', 'MMR Facility', 'MMR All-Intervention', 'MMR Facility-Intervention'])

    fig = alt.Chart(mmr_plot, title='Maternal Mortality Rate with Intervention').mark_line().encode(
        x=alt.X('x', title='% L2/3 to L4 Shift'),
        y=alt.Y('value', title='MMR'),
        color='variable')
    st.altair_chart(fig, use_container_width=True)

with right:
    with st.form('Test2'):
        transport_step = 100 / 19
        transport2 = st.slider(label="Transport Intervention - Select a value:",
                              min_value=0.0,
                              max_value=100.0,
                              value=0.0,
                              step=transport_step)
        quality_effect2 = st.slider(label='Quality care Intervention - Select a value:',
                                   min_value=0.0,
                                   max_value=75.0,
                                   step=0.5)
        anc_effect2 = st.slider(label='ANC Intervention - Select a value:',
                                   min_value=0.0,
                                   max_value=100.0,
                                   step=0.5)
        submitted2 = st.form_submit_button("Test Alternative Intervention")

    new_mmrall2, new_mmrfac2 = calc_effect(anc_effect2, quality_effect2)

    transport_index = int(transport2 / transport_step)
    mmr_plot2 = pd.concat([mmr_all.iloc[:, 0], mmr_fac.iloc[:, 0],
                          new_mmrall2.iloc[:, transport_index], new_mmrfac2.iloc[:, transport_index]],
                         axis=1)
    mmr_plot2.columns = ['MMR All', 'MMR Facility', 'MMR All-Intervention', 'MMR Facility-Intervention']
    mmr_plot2['x'] = np.linspace(0,90,4)
    mmr_plot2 = mmr_plot2.melt(id_vars=['x'], value_vars=[
        'MMR All', 'MMR Facility', 'MMR All-Intervention', 'MMR Facility-Intervention'])

    fig = alt.Chart(mmr_plot2, title='Maternal Mortality Rate with Alternative Intervention').mark_line().encode(
        x=alt.X('x', title='% L2/3 to L4 Shift'),
        y=alt.Y('value', title='MMR'),
        color='variable')
    st.altair_chart(fig, use_container_width=True)

def animate(i):
    mmr_plot = pd.concat([mmr_all.iloc[:, 0], mmr_fac.iloc[:, 0],
                          mmr_all.iloc[:, i], mmr_fac.iloc[:, i]],
                         axis=1)
    mmr_plot.columns = ['MMR All', 'MMR Facility', 'MMR All + Transport', 'MMR Facility + Transport']
    mmr_plot['x'] = np.linspace(0, 90, 4)
    mmr_plot = mmr_plot.melt(id_vars=['x'],
                             value_vars=['MMR All', 'MMR Facility', 'MMR All + Transport', 'MMR Facility + Transport'])
    fig = alt.Chart(mmr_plot,
                    title=f'Maternal Mortality Rate with {round(i*transport_step, 2)}% Transportation Intervention').mark_line().encode(
        x=alt.X('x', title='% L2/3 to L4 Shift'),
        y=alt.Y('value', title='MMR'),
        color='variable')
    return fig

fig2 = alt.Chart(mmr_plot.loc[mmr_plot['variable'].isin(['MMR All', 'MMR Facility']),:],
                 title='Maternal Mortality Rate with [    ]% Transportation Intervention').mark_line().encode(
    x=alt.X('x', title='% L2/3 to L4 Shift'),
    y=alt.Y('value', title='MMR'),
    color='variable')

st.subheader('**Animation of Transportation Intervention**')

left, middle, right = st.columns((2, 5, 2))
with middle:
    test = st.altair_chart(fig2, use_container_width=True)
    if st.button('Projection'):
        for i in range(0,19):
            fig3 = animate(i)
            test.altair_chart(fig3, use_container_width=True)
            time.sleep(0.5)