import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd
import altair as alt

### PARAMETERs ###
LB1s = [
    [1906, 1675, 2245, 118],
    [1799, 674, 1579, 0],
    [1711, 814, 1288, 77],
    [864, 1980, 2045, 99],
    [2949, 1407, 2248, 134],
    [1343, 1377, 73, 3997],
    [3789, 1201, 3349, 169],
    [2366, 2344, 1326, 122],
    [1760, 1370, 868, 81],
    [564, 1696, 2258, 91],
    [2184, 1853, 1579, 114],
    [2494, 1805, 1851, 124]
]

# SC.LB2s
LB2s = LB1s  # Assuming LB2s is identical to LB1s

# Capacity
Capacity = np.array([3972, 3108, 1020, 3168, 2412, 6888, 2580, 2136, 936, 2772, 1524, 4668])

# p_MM
p_MM = np.array([
    [0.0650, 1.0000, 0.0100, 0.1500],
    [0.0030, 0.0500, 0.0010, 0.0210]
])

# p_MM_scale_transfer
p_MM_scale_transfer = 3.0000

# n_CH.refer
n_CH_refer = np.array([
    [0, -53, 52, 19]
])

# n_CL.refer
n_CL_refer = np.array([
    [0, -12, 57, -31]
])

# q_CH.transfer_from
q_CH_transfer_from = np.array([
    [0.0000, 0.0000, 0.0000, 0.0024],
    [0.0000, 0.0000, 0.0458, 0.2906],
    [0.0000, 0.0337, 0.2302, 0.1932]
])

# q_CL.transfer_from
q_CL_transfer_from = np.array([
    [0.0000, 0.0000, 0.0000, 0.0024],
    [0.0000, 0.0000, 0.0530, 0.2778],
    [0.0000, 0.0361, 0.3072, 0.2291]
])

# LB_tot
LB_tot = [23729, 18196, 20709, 5126]

scale_CH_est = 0.035*0.5
scale_CL_est = 0.035*0.5

### MODEL PERTINENT ###
n_months = 12

SC = {
    'n': 12,
    'LB1s': {
        0: np.zeros((n_months, 4)),
        1: np.zeros((n_months, 4)),
        2: np.zeros((n_months, 4)),
        3: np.zeros((n_months, 4)),
        4: np.zeros((n_months, 4)),
        5: np.zeros((n_months, 4)),
        6: np.zeros((n_months, 4)),
        7: np.zeros((n_months, 4)),
        8: np.zeros((n_months, 4)),
        9: np.zeros((n_months, 4)),
        10: np.zeros((n_months, 4)),
        11: np.zeros((n_months, 4)),
    },
    'LB2s': {
        0: np.zeros((n_months, 4)),
        1: np.zeros((n_months, 4)),
        2: np.zeros((n_months, 4)),
        3: np.zeros((n_months, 4)),
        4: np.zeros((n_months, 4)),
        5: np.zeros((n_months, 4)),
        6: np.zeros((n_months, 4)),
        7: np.zeros((n_months, 4)),
        8: np.zeros((n_months, 4)),
        9: np.zeros((n_months, 4)),
        10: np.zeros((n_months, 4)),
        11: np.zeros((n_months, 4)),
    }
}

for i in range(n_months):
    SC['LB1s'][i][0, :] = LB1s[i]
    SC['LB2s'][i][0, :] = LB1s[i]

### CHOOSE INTERVENTIONS ###
st.header('SDR interventions')
options = {
    'Off': 0,
    'On': 1
}

col1, col2, col3, col4 = st.columns(4)
with col1:
    CHVint = st.selectbox('CHV intervention', list(options.keys()))
    flag_CHV = options[CHVint]
    n_months_push_back = st.slider('Number of months to push back', min_value=1, max_value=36, step=1, value=2)
    b_push_back = st.slider('Coefficient of push back', min_value=1, max_value=10, step=1, value=5)
    CHV_b = st.slider('CHV effect on L4 prob', min_value=0.00, max_value=0.10, step=0.01, value=0.02)
with col2:
    ANCint = st.selectbox('ANC intervention', list(options.keys()))
    flag_ANC= options[ANCint]
with col3:
    Referint = st.selectbox('Referral intervention', list(options.keys()))
    flag_refer = options[Referint]
with col4:
    Transint = st.selectbox('Transfer intervention', list(options.keys()))
    flag_trans = options[Transint]

INT = {
    'CHV': {
        #'n_months_push_back': 2,
        #'b_push_back': 5,
        'n_months_push_back': n_months_push_back,
        'b_push_back': b_push_back,
        'L4': {
            'b': 0
        }
    },
    'ANC': {
        'b0': 0,
        # 'bs': 1 - (1 - INT['ANC']['b0']) ** np.minimum(t, 12),
        'bs': [],
        'b': None
    },
    'refer': {
        'b0': 0,
        # 'bs': INT['refer']['b0'] * (t - 1),
        'bs': [],
        'b': None
    },
    'trans': {
        'b0': 0,
        'bs': np.ones(n_months),
        'b': None
    }
}

Capacity_ratio1 = np.zeros((n_months, SC['n']))
Capacity_ratio2 = np.zeros((n_months, SC['n']))
Push_back = np.zeros((n_months, SC['n']))
n_MM = np.zeros((n_months, 4, 2))  # number of months, for each hospital level, with and without pushback

t = n_months

#flag_CHV = 1
#flag_ANC = 1
#flag_refer = 1
#flag_trans = 0

# Update dictionary values based on flags
if flag_CHV:
    #INT['CHV']['L4']['b'] = 0.02
    INT['CHV']['L4']['b'] = CHV_b
if flag_ANC:
    INT['ANC']['b0'] = 0.03

if flag_refer:
    INT['refer']['b0'] = 0.1

if flag_trans:
    INT['trans']['b0'] = 1
    INT['trans']['bs'] = INT['trans']['b0'] * (t < n_months / 2)


def f_INT_LB_effect(SC_LB_previous, INT, flag_push_back, i, Capacity_ratio):
    INT_b = INT['CHV']['L4']['b']
    push_back = 0
    if flag_push_back:
        i_months = range(max(1, i - INT['CHV']['n_months_push_back']),
                         i - 1)  # months going from -n_months to -1 months
        mean_capacity_ratio = np.mean(Capacity_ratio[i_months])  # average over those months
        push_back = max(0, mean_capacity_ratio - 1)  # zero = no pushback
        scale = np.exp(-push_back * INT['CHV']['b_push_back'])  # use exponent for the scale
        INT_b = INT_b * scale  # reduce the CHV effect

    effect = np.array([np.exp(-INT_b * np.array([1, 1])), 1 + INT_b * np.array([1, 1])]).reshape(1,
                                                                                                 4)  # effect of intervention on LB
    SC_LB = SC_LB_previous * effect  # new numbers of LB
    SC_LB = SC_LB / np.sum(SC_LB) * np.sum(SC_LB_previous)  # sum should equal the prior sum

    return SC_LB, push_back


# DETAILS
def f_MM(LB_tot, SDR_multiplier_0, scale_CH_est, n_C0_refer, q_C0_transfer_from, p_MM, p_MM_scale_transfer, INT):
    q_C0_transfer_from = q_C0_transfer_from * INT['trans']['b0']
    n_MM = np.zeros(4)
    LB_tot_k = LB_tot * SDR_multiplier_0
    LB_tot_k = LB_tot_k * np.sum(LB_tot) / np.sum(LB_tot_k)
    SDR_multiplier_k = LB_tot_k / LB_tot
    n_refer_k = n_C0_refer * SDR_multiplier_k
    n_initial_est_k = LB_tot_k * scale_CH_est
    n_refer_k = n_refer_k * (1 + INT['refer']['b'])
    n_initial_C = n_refer_k + n_initial_est_k
    n_transfer_from_k = q_C0_transfer_from * n_initial_C[0:3]
    n_transfer_out = np.append(np.sum(n_transfer_from_k, axis=1), 0)
    n_transfer_in = np.sum(n_transfer_from_k, axis=0)
    n_not_transfer = n_initial_C - n_transfer_out
    n_MM = n_MM + np.maximum(0, n_not_transfer) * p_MM
    n_MM = n_MM + n_transfer_in * p_MM * p_MM_scale_transfer

    return n_MM


for i in range(t):
    ### OKAY ###
    LB1_tot_i = np.zeros(4)
    LB2_tot_i = np.zeros(4)
    INT['refer']['bs'].append(INT['refer']['b0'] * (t - 1))
    INT['refer']['b'] = INT['refer']['bs'][i]

    INT['ANC']['bs'].append(1 - (1 - INT['ANC']['b0']) ** np.minimum(t, 12))
    INT['ANC']['b'] = INT['ANC']['bs'][i]

    INT['trans']['b'] = INT['trans']['bs'][i]
    scale_CH_est_i = scale_CH_est * (1 - INT['ANC']['b'])
    scale_CL_est_i = scale_CL_est * (1 - INT['ANC']['b'])
    ### OKAY ###

    for j in range(SC['n']):  # for each sub county
        if i > 0:  # skip time 1
            SC['LB1s'][j][i, :], NA = f_INT_LB_effect(SC['LB1s'][j][i - 1, :], INT, False, i, Capacity_ratio1[:, j])
            # compute pushback (overcapacity => reduced CHV effect)
            SC['LB2s'][j][i, :], Push_back[i, j] = f_INT_LB_effect(SC['LB2s'][j][i - 1, :], INT, True, i,
                                                                   Capacity_ratio2[:, j])

        Capacity_ratio1[i, j] = np.sum(SC['LB1s'][j][i, 2:4]) / Capacity[j]  # LB/capacity (no pushback)
        Capacity_ratio2[i, j] = np.sum(SC['LB2s'][j][i, 2:4]) / Capacity[j]  # (pushback)
        LB1_tot_i += SC['LB1s'][j][i, :]
        LB2_tot_i += SC['LB2s'][j][i, :]

    SDR_multiplier_1 = LB1_tot_i / LB_tot
    SDR_multiplier_2 = LB2_tot_i / LB_tot
    n_MM1_H = f_MM(LB_tot, SDR_multiplier_1, scale_CH_est_i, n_CH_refer, q_CH_transfer_from, p_MM[1],
                   p_MM_scale_transfer, INT)
    n_MM2_H = f_MM(LB_tot, SDR_multiplier_2, scale_CH_est_i, n_CH_refer, q_CH_transfer_from, p_MM[1],
                   p_MM_scale_transfer, INT)
    n_MM1_L = f_MM(LB_tot, SDR_multiplier_1, scale_CL_est_i, n_CL_refer, q_CL_transfer_from, p_MM[0],
                   p_MM_scale_transfer, INT)
    n_MM2_L = f_MM(LB_tot, SDR_multiplier_2, scale_CL_est_i, n_CL_refer, q_CL_transfer_from, p_MM[0],
                   p_MM_scale_transfer, INT)

    n_MM[i, :, 0] = n_MM1_H + n_MM1_L
    n_MM[i, :, 1] = n_MM2_H + n_MM2_L

##Transfer n_MM into dataframes
dfs = []
for j in range(2):
    facility_levels = [f'{i}' for i in range(n_MM[:, :, j].T.shape[0])]
    months = [i + 1 for i in range(n_MM[:, :, j].T.shape[1])]

    data = {
        "level": [level for level in facility_levels for _ in range(n_MM[:, :, j].T.shape[1])],
        "month": [month for _ in range(n_MM[:, :, j].T.shape[0]) for month in months],
        "value": n_MM[:, :, j].T.flatten()  # You can name this column as needed
    }

    data["pushback"] = j
    df = pd.DataFrame(data)
    dfs.append(df)

dfs = pd.concat(dfs, ignore_index=True)
dfs['level'] = dfs['level'].astype('category')

###Plot #Death by subcounty
# Filter the DataFrame to create subsets for pushback = 0 and pushback = 1
df_pushback_0 = dfs[dfs['pushback'] == 0]
df_pushback_1 = dfs[dfs['pushback'] == 1]

col1, col2 = st.columns(2)
with col1:
    chart_pushback_0 = (
        alt.Chart(
            data=df_pushback_0,
            title="Pushback 0",
        )
        .mark_line()
        .encode(
            x=alt.X("month", axis=alt.Axis(title="Month")),
            y=alt.Y("value", axis=alt.Axis(title="Deaths")).scale(domain=(0, 400)),
            color=alt.Color("level:N").title("Level")
        )
    )

    st.altair_chart(chart_pushback_0)

with col2:
    chart_pushback_1 = (
        alt.Chart(
            data=df_pushback_1,
            title="Pushback 1",
        )
        .mark_line()
        .encode(
            x=alt.X("month", axis=alt.Axis(title="Month")),
            y=alt.Y("value", axis=alt.Axis(title="Deaths")).scale(domain=(0, 400)),
            color=alt.Color("level:N").title("Level")
        )
    )

    st.altair_chart(chart_pushback_1)

# ##Plot #Death by subcounty
# num_rows, num_cols = 1, 2
#
# # Create a grid of subplots
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))
#
# # Iterate over j from 0 to 1 (two subplots)
# for j in range(2):
#     # Replace 0 with j to access different data
#     MM = n_MM[:, :, j].T
#
#     # Plot the data in the current subplot
#     for i in range(4):
#         axes[j].plot(range(12), MM[i, :])
#     axes[j].set_title(f'Pushback {j}')
#     axes[j].set_xlabel('Months')
#     axes[j].set_ylabel('Deaths')
#
# # Adjust layout
# plt.tight_layout()
#
# # Show the plots
# #plt.show()
# st.pyplot(fig)
#
# ###Plot births by subcounty
#
# num_rows, num_cols = 4, 3
#
# # Create a grid of subplots
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
#
# # Iterate over j from 0 to 11
# for j in range(num_rows * num_cols):
# # Replace 7 with j to access different LB1s
#     y_values = SC['LB1s'][j]
#     y_values = y_values.T
#
#     # Calculate the row and column indices for the current subplot
#     row, col = divmod(j, num_cols)
#
#     # Plot the data in the current subplot
#     for i in range(4):
#         axes[row, col].plot(range(12), y_values[i])
#
#     axes[row, col].set_title(f'Subcounty {j + 1}')
#     axes[row, col].set_xlabel('Months')
#     axes[row, col].set_ylabel('# of Births')
#
# # Adjust layout
# plt.tight_layout()
#
# # Show the plots
#     #plt.show()
# st.pyplot(fig)
#
#
