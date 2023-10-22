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


# MODEL PARAMETERS
n_months = 36  # number of months to run the model
t = np.arange(1, n_months + 1)  # time
CHV_ES = 0.01  # effect size of CHV intervention
CHV_L4_b = 0.02  # influence of CHV on increasing probability of L4
CHV_ttl = ["Home", "L2/3", "L4", "L5", "Capacity ratio, no pushback", "Capacity ratio, pushback"]
n_sc = 12
# TABLE SC
# Each row of SC corresponds to a sub county j
# SC has a column of data SC.LB1s in which SC.LB1s[j] = number of LB in [H, L2, L3, L4/5]

SC_LB1s = np.zeros((n_sc, n_months,4))

SC_LB1s[:,0,:] = np.array([
    [2088, 723, 952, 2181],
    [1799, 40, 634, 1579],
    [1830, 121, 693, 1246],
    [1017, 290, 1690, 1991],
    [3156, 224, 1183, 2175],
    [288, 318, 1059, 5125],
    [4050, 560, 641, 3257],
    [2554, 932, 1412, 1260],
    [1885, 766, 604, 824],
    [18, 1214, 482, 2895],
    [2360, 213, 1640, 1517],
    [2686, 340, 1465, 1783]
])

SC0_LB0s = SC_LB1s.copy()
SC0_L4_capacity1 = np.array([
    [9024],
    [3108],
    [1896],
    [4740],
    [3756],
    [7044],
    [2940],
    [2304],
    [1020],
    [2808],
    [1560],
    [4668]
])

SC_LB2s = SC_LB1s.copy()
SC_LB1_2022 = SC_LB1s.copy()
n_months_push_back = 2  # number of previous months to compute capacity ratio
b_push_back = 5  # push back effect size
Capacity_ratio1 = np.zeros((n_months, 12))
Capacity_ratio2 = np.zeros((n_months, 12))
Push_back = np.zeros((n_months, 12))

# COMPUTE EFFECT OF CHV
# Scale the number of mothers in [H, L2, L3, L4/5] using this vector of 4 values
effect_no_pushback = np.exp(-CHV_L4_b * np.array([1, 1, 1, 1])) * (1 + CHV_L4_b)

for j in range(n_sc):  # for each sub county
    for i in range(n_months):
        if i > 0:  # skip sub county 1 (because it already has high L4)
            LB1s_prior = SC_LB1s[j][i - 1]  # number of LBs in the prior time step
            LB2s_prior = SC_LB2s[j][i - 1]  # LB1s = no pushback, LB2s = pushback
            LB1s_current = LB1s_prior * effect_no_pushback  # new numbers of LB without pushback
            SC_LB1s[j][i, :] = LB1s_current / np.sum(LB1s_current) * np.sum(LB1s_prior)

            # Compute pushback (overcapacity => reduced CHV effect)
            i_months = np.arange(max(0, i - n_months_push_back), i)  # months going from -n_months to -1 months
            mean_capacity_ratio = np.mean(Capacity_ratio2[i_months, j])  # average over those months
            Push_back[i, j] = max(0, mean_capacity_ratio - 1)  # zero = no pushback
            scale = np.exp(-Push_back[i, j] * b_push_back)  # use exponent for the scale
            effect_pushback = np.array([np.exp(-CHV_L4_b*np.array([1,1,1])*scale)])
            effect_pushback = np.append(effect_pushback, 1+CHV_L4_b*scale)
            LB2s_current = LB2s_prior * effect_pushback  # new numbers of LB with pushback
            SC_LB2s[j][i, :] = LB2s_current / np.sum(LB2s_current) * np.sum(LB2s_prior)

        Capacity_ratio1[i, j] = SC_LB1s[j][i, 3] / SC0_L4_capacity1[j]  # LB at L4 / capacity
        Capacity_ratio2[i, j] = SC_LB2s[j][i, 3] / SC0_L4_capacity1[j]  # > 1 => over capacity

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
for j in range(n_sc):
    ax1.plot(range(n_months), Capacity_ratio1[:,j])
    ax1.set_title('Capacity ratio 1')
for j in range(n_sc):
    ax2.plot(range(n_months), Capacity_ratio2[:,j])
    ax2.set_title('Capacity ratio 2')