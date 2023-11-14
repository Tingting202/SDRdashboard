import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
from sympy import symbols, Eq, solve
import plotly.express as px
import pandas as pd
import altair as alt

### Sliders ###
st.header('SDR interventions')
options = {
    'Off': 0,
    'On': 1
}

col1, col2, col3, col4 = st.columns(4)
with col1:
    CHVint = st.selectbox('CHV', list(options.keys()))
    flag_CHV = options[CHVint]
    # n_months_push_back = st.slider('Number of months to push back', min_value=1, max_value=36, step=1, value=2)
    # b_push_back = st.slider('Coefficient of push back', min_value=1, max_value=10, step=1, value=5)
    CHV_b = st.slider('CHV effect on L4 prob', min_value=0.00, max_value=0.10, step=0.01, value=0.02)
with col2:
    subint = st.selectbox('Subcounty', list(options.keys()))
    flag_sub = options[subint]
    ANCint = st.selectbox('ANC', list(options.keys()))
    flag_ANC= options[ANCint]
    ANC2int = st.selectbox('ANC effect on complications', list(options.keys()))
    flag_ANC2= options[ANC2int]
with col3:
    Referint = st.selectbox('Referral intervention', list(options.keys()))
    flag_refer = options[Referint]
    referrate = st.slider('refer rate', min_value=0.0, max_value=1.0, step=0.1, value=0.5)
    Transint = st.selectbox('Transfer intervention', list(options.keys()))
    flag_trans = options[Transint]
with col4:
    int1 = st.selectbox('Obstetric drape', list(options.keys()))
    flag_int1 = options[int1]
    int2 = st.selectbox('IV Iron', list(options.keys()))
    flag_int2 = options[int2]
    int4 = st.selectbox('Antenatal corticosteroids', list(options.keys()))
    flag_int4 = options[int4]
    int3 = st.selectbox('Ultrasound', list(options.keys()))
    flag_int3 = options[int3]
    diagnosisrate = st.slider('Ultrasound diagnosis rate', min_value=0.0, max_value=1.0, step=0.1, value=0.5)

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

CHs = np.array([
    [0, -8, 9, 1],
    [0, -3, 6, 0],
    [0, -4, 5, 1],
    [0, -10, 7, 0],
    [0, -7, 9, 2],
    [0, -7, 1, 33],
    [0, -7, 12, 2],
    [0, -13, 6, 1],
    [0, -7, 4, 1],
    [0, -9, 8, 1],
    [0, -9, 6, 1],
    [0, -9, 7, 1]
])

CLs = np.array([
    [0, -17, 8, 4],
    [0, -7, 5, 0],
    [0, -8, 4, 3],
    [0, -21, 6, 3],
    [0, -15, 8, 5],
    [0, -14, 1, 136],
    [0, -13, 10, 6],
    [0, -25, 5, 4],
    [0, -14, 3, 3],
    [0, -18, 7, 3],
    [0, -19, 5, 4],
    [0, -19, 6, 4]
])

# Convert the list of lists to a NumPy array

# SC.LB2s
LB2s = LB1s  # Assuming LB2s is identical to LB1s

# Capacity
Capacity = np.array([3972, 3108, 1020, 3168, 2412, 6888, 2580, 2136, 936, 2772, 1524, 4668])

# p_MM
p_MM = np.array([
    [0.0650, 0.01,   0.003, 0.0010],
    [1,      0.1500, 0.05,  0.0210]
])

# p_MM_scale_transfer
p_MM_scale_transfer = 3.0000

# number of high complications referred
n_CH_refer = np.array([
    [0, -93, 80, 42]
])

# number of low complications referred
n_CL_refer = np.array([
    [0, -190, 68, 174]
])

# q_CH.transfer_from
q_CH_transfer_from = np.array([
    [0.0000, 0.0024, 0.0289, 0.0241],
    [0.0000, 0.0000, 0.2533, 0.1822],
    [0.0000, 0.0000, 0.0000, 0.1833]
])

# q_CL.transfer_from
q_CL_transfer_from = np.array([
    [0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000]
])

# LB_tot
LB_tot = [23729, 18196, 20709, 5126]

scale_CH_est = 0.035*0.5
scale_CL_est = 0.035*0.5
scale_N = 16

### MODEL PERTINENT ###
n_months = 36
t = np.arange(n_months)

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

n_CHs = {
    'refer': {
        0: np.zeros((4)),
        1: np.zeros((4)),
        2: np.zeros((4)),
        3: np.zeros((4)),
        4: np.zeros((4)),
        5: np.zeros((4)),
        6: np.zeros((4)),
        7: np.zeros((4)),
        8: np.zeros((4)),
        9: np.zeros((4)),
        10: np.zeros((4)),
        11: np.zeros((4)),
    }
}

n_CLs = {
    'refer': {
        0: np.zeros((4)),
        1: np.zeros((4)),
        2: np.zeros((4)),
        3: np.zeros((4)),
        4: np.zeros((4)),
        5: np.zeros((4)),
        6: np.zeros((4)),
        7: np.zeros((4)),
        8: np.zeros((4)),
        9: np.zeros((4)),
        10: np.zeros((4)),
        11: np.zeros((4)),
    }
}

for i in range(SC['n']):
    SC['LB1s'][i][0, :] = LB1s[i]
    SC['LB2s'][i][0, :] = LB1s[i]

for i in range(SC['n']):
    n_CHs['refer'][i] = CHs[i]
    n_CLs['refer'][i] = CLs[i]

INT = {
    'CHV': {
        'n_months_push_back': 2,
        'b_push_back': 5,
        'L4': {
            'b': 0
        }
    },
    'ANC': {
        'inc': 0.585,
        'b0': 0,
        'bs': None,
        'b': None,
        'ANC_PPH': {
            'b': 0.5 * 0.017 * (1 - 0.812) / scale_CH_est
        },
        'ANC_msepsis': {
            'b': None
        },
        'ANC_eclampsia': {
            'b': None
        },
        'ANC_obs_labor': {
            'b': 0.5 * 0.1 * (1 - 0.62) / scale_CH_est
        },
        'ANC_rup_uterus': {
            'b': None
        },
        'ANC_preterm': {
            'b': None
        },
    },
    'refer': {
        'b0': 0,
        'bs': None,
        'b': None,
        'refer_rate': 0  # 0
    },
    'trans': {
        'b0': 0,
        'bs': np.ones(n_months),
        'b': None
    },
    'comp': {
        'PPH': {
            'inc': 0.017 * 0.5,
            'b0': 1,
            'b': 1
        },
        'APH': {
            'inc': 0.0039 * 0.5,
            'b0': 1,
            'b': 1
        },
        'msepsis': {
            'inc': 0.00159 * 0.5
        },
        'eclampsia': {
            'inc': 0.0046 * 0.5
        },
        'obs_labor': {
            'inc': 0.01
        },
        'rup_uterus': {
            'inc': 0.0005
        },
        'Anemia': {
            'inc': 0.25
        },
        'Anemia_PPH': {
            'b': 1,
        },
        'Anemia_APH': {
            'b': 1
        },
        'PP': {
            'inc': 0.0027 * 0.5,
            'b0': 1,
            'b': 1
        },
        'PP_PPH': {
            'b': 1
        }

    },
    'mort': {
        'red': 0,
        'b': 0
    },
    'neo': {
        'preterm': 1
    },
    'ultrasound': {
        'diagnosis_rate': 0
    }
}

Capacity_ratio1 = np.zeros((n_months, SC['n']))
Capacity_ratio2 = np.zeros((n_months, SC['n']))
Push_back = np.zeros((n_months, SC['n']))
n_MM = np.zeros((n_months, 4, 2))  # number of months, for each hospital level, with and without pushback
n_nM = np.zeros((n_months, 4, 2))  # neonatal mortality
n_MMs = np.zeros((n_months, 4, 12, 2))
p_MMs = np.zeros((n_months, 4, 12, 2))

####Referral due to Ultrasound at whole county level#######
# Calculate the number of mothers with placenta previa and placental abruption
N_pp = np.array(LB_tot) * 0.0027  # placenta previa
N_pa = np.array(LB_tot) * 0.01  # placental abruption
N_pph_bypp = N_pp * 0.0199
N_pph_bypa = N_pa * 0.182
N_preterm_bypp = N_pp * 0.559
N_preterm_bypa = N_pa * 0.396
N_preterm_byother = np.array(LB_tot) * 0.023523 - (N_preterm_bypp + N_preterm_bypa)
# Calculate the number of PPH, APH, and preterm from our facility data
N_pph_facility = np.array(LB_tot) * 0.017
N_preterm_facility = np.array(LB_tot) * 0.023523

# Calculate the number of pph, aph, and preterm can be detected due to ultrasound diagnosis
N_pph_detected = N_pph_bypp * 0.875 + N_pph_bypa * 0.57
N_preterm_detected = N_preterm_bypp * 0.875 + N_preterm_bypa * 0.57 + N_preterm_byother * 0.99
# Assume all detected pph in L2/L3 will be referred to L4 or L5
N_pph_refered = np.array([0, -N_pph_detected[1], N_pph_detected[1] * 0.5, N_pph_detected[1] * 0.5])
N_preterm_refered = np.array([0, -N_preterm_detected[1], N_preterm_detected[1] * 0.5, N_preterm_detected[1] * 0.5])

### CHOOSE INTERVENTIONS ###
def comp_reduction(oddsratio, p_expose, p_comp, int_expose):
    """given odds ratio and exposure, return change in complications due to change in exposure"""

    x, y = symbols('x y')
    eq1 = Eq(x / (1 - x) / (y / (1 - y)) - oddsratio, 0)
    eq2 = Eq(p_comp - p_expose * x - (1 - p_expose) * y, 0)
    solution = solve((eq1, eq2), (x, y))[0]

    base_comp = solution[0] * p_expose + solution[1] * (1 - p_expose)
    new_comp = solution[0] * p_expose * int_expose + solution[1] * (1 - p_expose * int_expose)

    change = (scale_CH_est - base_comp + new_comp) / scale_CH_est
    return change

def f_INT_LB_effect(SC_LB_previous, INT, flag_push_back, i, Capacity_ratio):
    """calculates pushback effect on change in live births """
    INT_b = INT['CHV']['L4']['b']

    if flag_push_back:
        i_months = range(max(0, i-INT['CHV']['n_months_push_back']), i)  # months going from -n_months to -1 months
        mean_capacity_ratio = np.mean(Capacity_ratio[i_months])  # average over those months
        push_back = max(0, mean_capacity_ratio - 1)  # zero = no pushback
        scale = np.exp(-push_back * INT['CHV']['b_push_back'])  # use exponent for the scale
        INT_b = INT_b * scale  # reduce the CHV effect

        effect = np.array([np.exp(-INT_b * np.array([1, 1])), 1 + INT_b * np.array([1, 1])]).reshape(1,4)  # effect of intervention on LB
        SC_LB = SC_LB_previous * effect  # new numbers of LB
        SC_LB = SC_LB / np.sum(SC_LB) * np.sum(SC_LB_previous)  # sum should equal the prior sum
        return SC_LB, push_back

    else:
        effect = np.array([np.exp(-INT_b * np.array([1, 1])), 1 + INT_b * np.array([1, 1])]).reshape(1,4)  # effect of intervention on LB
        SC_LB = SC_LB_previous * effect  # new numbers of LB
        SC_LB = SC_LB / np.sum(SC_LB) * np.sum(SC_LB_previous)  # sum should equal the prior sum
        return SC_LB

# DETAILS
def f_MM(LB_tot, SDR_multiplier_0, scale_CH_est, n_C0_refer, q_C0_transfer_from, p_MM, p_MM_scale_transfer, INT):
    """calculates mortality """
    q_C0_transfer_from = q_C0_transfer_from * INT['trans']['b']
    n_MM = np.zeros(4)
    LB_tot_k = LB_tot * SDR_multiplier_0
    LB_tot_k = LB_tot_k * np.sum(LB_tot) / np.sum(LB_tot_k)
    SDR_multiplier_k = LB_tot_k / LB_tot
    n_refer_k = (n_C0_refer * INT['refer']['refer_rate'] + N_pph_refered * INT['ultrasound']['diagnosis_rate']) * SDR_multiplier_k                        # number referred
    n_initial_est_k = LB_tot_k * scale_CH_est # number of complications original
    n_refer_k = n_refer_k * (1+INT['refer']['b'])                            # change in referral due to intervention
    n_initial_C = n_refer_k + n_initial_est_k                                # initial complications at locations
    n_initial_C = n_initial_C.flatten()
    n_transfer_from_k = np.diag(n_initial_C[0:3]) @ q_C0_transfer_from       # number transferred given original complications and transfer probabilities
    n_transfer_out = np.append(np.sum(n_transfer_from_k, axis=1), 0)         # transferred out
    n_transfer_in = np.sum(n_transfer_from_k, axis=0)                        # transferred in
    n_not_transfer = n_initial_C - n_transfer_out                            # not transferred
    n_MM = n_MM + np.maximum(0, n_not_transfer) * p_MM                       # considering transfer, not transfer, effect on mortality
    n_MM = n_MM + n_transfer_in * p_MM * p_MM_scale_transfer

    n_nM = n_MM * scale_N * INT['neo']['preterm']
    # n_nM = max(0, n_nM - INT['mort']['red'] * LB_tot_k)

    stats = [q_C0_transfer_from, n_initial_est_k, n_refer_k, n_initial_C]

    return n_MM, n_nM, stats


comps = ['Low PPH', 'High PPH', 'Neonatal Deaths', 'Maternal Deaths']
DALYs = [0.114, 0.324, 1, 0.54]
DALY_dict = {x: 0 for x in comps}

def run_model(flags):
    outcomes_dict = {x: 0 for x in comps}

    flag_sub = flags[0]
    flag_CHV = flags[1]
    flag_ANC = flags[2]
    flag_ANC2 = flags[3]
    flag_refer = flags[4]
    flag_trans = flags[5]
    flag_int1 = flags[6]  # Obstetric drape
    flag_int2 = flags[7]  # Anemia reduction through IV Iron
    flag_int3 = flags[8]  # ultrasound
    flag_int4 = flags[9]  # antenatal corticosteroids - reduction of neonatal mortality

    # Update dictionary values based on flags
    if flag_CHV:
        INT['CHV']['L4']['b'] = CHV_b
    else:
        INT['CHV']['L4']['b'] = 0
    if flag_ANC:
        INT['ANC']['b0'] = 0.03
    else:
        INT['ANC']['b0'] = 0
    if flag_refer:
        INT['refer']['b0'] = 0.1
        INT['refer']['refer_rate'] = referrate
    else:
        INT['refer']['b0'] = 0
        INT['refer']['refer_rate'] = 0
    if flag_trans:
        INT['trans']['b0'] = 1
        INT['trans']['bs'] = INT['trans']['b0'] * (t < n_months / 2)
    else:
        INT['trans']['bs'] = np.ones(n_months)
    if flag_int1:
        INT['comp']['PPH']['b'] = 1 - 0.6 * INT['comp']['PPH']['inc'] / (scale_CH_est)  # difference from original
    else:
        INT['comp']['PPH']['b'] = 1
    if flag_int2:
        INT['comp']['Anemia_PPH']['b'] = comp_reduction(3.54, INT['comp']['Anemia']['inc'], INT['comp']['PPH']['inc'],
                                                        0.3)
        INT['comp']['Anemia_APH']['b'] = comp_reduction(1.522, INT['comp']['Anemia']['inc'], INT['comp']['APH']['inc'],
                                                        0.3)
    else:
        INT['comp']['Anemia_PPH']['b'] = 1
        INT['comp']['Anemia_APH']['b'] = 1
    if flag_int3:
        INT['ultrasound']['diagnosis_rate'] = diagnosisrate
    else:
        INT['ultrasound']['diagnosis_rate'] = 0
    if flag_int4:
        if flag_ANC:
            INT['neo'][
                'preterm'] = 0.788  # percent reduction in neonatal deaths overall due to 16% reduction in neonatal deaths as a result of preterm labor
    else:
        INT['neo']['preterm'] = 1

    INT['ANC']['bs'] = 1 - (1 - INT['ANC']['b0']) ** np.minimum(t, 12)
    INT['refer']['bs'] = INT['refer']['b0'] * (t - 1)

    if flag_ANC:
        INT['ANC']['ANC_msepsis']['b'] = np.array(
            [comp_reduction(4, INT['comp']['msepsis']['inc'], INT['ANC']['inc'], i) for i in
             (1 - INT['ANC']['bs'] / (INT['ANC']['b0'] * 12))])
        INT['ANC']['ANC_eclampsia']['b'] = np.array(
            [comp_reduction(1.41, INT['comp']['eclampsia']['inc'], INT['ANC']['inc'], i) for i in
             (1 - INT['ANC']['bs'] / (INT['ANC']['b0'] * 12))])
        INT['ANC']['ANC_rup_uterus']['b'] = np.array(
            [comp_reduction(6.29, INT['comp']['rup_uterus']['inc'], INT['ANC']['inc'], i) for i in
             (1 - INT['ANC']['bs'] / (INT['ANC']['b0'] * 12))])

    if flag_ANC2:
        flag_ANC = 1
        INT['ANC']['bs'] = INT['ANC']['bs'] * (
                    INT['ANC']['ANC_PPH']['b'] * INT['ANC']['ANC_msepsis']['b'] * INT['ANC']['ANC_eclampsia']['b'] *
                    INT['ANC']['ANC_obs_labor']['b'] * INT['ANC']['ANC_rup_uterus']['b'])
        INT['mort']['b'] = INT['ANC']['ANC_preterm']['b']

    n_stats_H1 = []
    n_stats_H2 = []
    n_stats_L1 = []
    n_stats_L2 = []

    for i in t:
        LB1_tot_i = np.zeros(4)
        LB2_tot_i = np.zeros(4)
        INT['refer']['b'] = INT['refer']['bs'][i]
        INT['ANC']['b'] = INT['ANC']['bs'][i]
        INT['trans']['b'] = INT['trans']['bs'][i]
        scale_CH_est_i = scale_CH_est * (1 - INT['ANC']['b']) * INT['comp']['Anemia_PPH']['b'] * \
                         INT['comp']['Anemia_APH']['b'] * INT['comp']['PP_PPH']['b']
        scale_CL_est_i = scale_CL_est * (1 - INT['ANC']['b']) * INT['comp']['Anemia_PPH']['b'] * \
                         INT['comp']['Anemia_APH']['b']

        for j in range(SC['n']):  # for each sub county
            if i > 0:  # skip time 1
                SC['LB1s'][j][i, :] = f_INT_LB_effect(SC['LB1s'][j][i - 1, :], INT, False, i, Capacity_ratio1[:, j])
                # compute pushback (overcapacity => reduced CHV effect)
                SC['LB2s'][j][i, :], Push_back[i, j] = f_INT_LB_effect(SC['LB2s'][j][i - 1, :], INT, True, i,
                                                                       Capacity_ratio2[:, j])

                if flag_sub:
                    LB_tot_j = np.maximum(SC['LB1s'][j][0, :], 1)  # (zero LB in L5 for sub 2)
                    SDR_multiplier_1_j = SC['LB1s'][j][i, :] / LB_tot_j  # ratio of sums = 1
                    SDR_multiplier_2_j = SC['LB2s'][j][i, :] / LB_tot_j

                    n_MM1s_H, _, _ = f_MM(LB_tot_j, SDR_multiplier_1_j, scale_CH_est_i, n_CHs['refer'][j],
                                          q_CH_transfer_from, p_MM[1], p_MM_scale_transfer, INT)
                    n_MM2s_H, _, _ = f_MM(LB_tot_j, SDR_multiplier_2_j, scale_CH_est_i, n_CHs['refer'][j],
                                          q_CH_transfer_from, p_MM[1], p_MM_scale_transfer, INT)
                    n_MM1s_L, _, _ = f_MM(LB_tot_j, SDR_multiplier_1_j, scale_CL_est_i, n_CLs['refer'][j],
                                          q_CL_transfer_from, p_MM[0], p_MM_scale_transfer, INT)
                    n_MM2s_L, _, _ = f_MM(LB_tot_j, SDR_multiplier_2_j, scale_CL_est_i, n_CLs['refer'][j],
                                          q_CL_transfer_from, p_MM[0], p_MM_scale_transfer, INT)

                    n_MMs[i, :, j, 0] = n_MM1s_H + n_MM1s_L
                    n_MMs[i, :, j, 1] = n_MM2s_H + n_MM2s_L
                    p_MMs[i, :, j, :] = n_MMs[i, :, j, :] / LB_tot_j[:, None]

            Capacity_ratio1[i, j] = np.sum(SC['LB1s'][j][i, 2:4]) / Capacity[j]  # LB/capacity (no pushback)
            Capacity_ratio2[i, j] = np.sum(SC['LB2s'][j][i, 2:4]) / Capacity[j]  # (pushback)
            LB1_tot_i += SC['LB1s'][j][i, :]
            LB2_tot_i += SC['LB2s'][j][i, :]

        SDR_multiplier_1 = LB1_tot_i / LB_tot
        SDR_multiplier_2 = LB2_tot_i / LB_tot
        n_MM1_H, n_nM1_H, stats_H1 = f_MM(LB_tot, SDR_multiplier_1, scale_CH_est_i, n_CH_refer, q_CH_transfer_from,
                                          p_MM[1], p_MM_scale_transfer, INT)
        n_MM2_H, n_nM2_H, stats_H2 = f_MM(LB_tot, SDR_multiplier_2, scale_CH_est_i, n_CH_refer, q_CH_transfer_from,
                                          p_MM[1], p_MM_scale_transfer, INT)
        n_MM1_L, n_nM1_L, stats_L1 = f_MM(LB_tot, SDR_multiplier_1, scale_CL_est_i, n_CL_refer, q_CL_transfer_from,
                                          p_MM[0], p_MM_scale_transfer, INT)
        n_MM2_L, n_nM2_L, stats_L2 = f_MM(LB_tot, SDR_multiplier_2, scale_CL_est_i, n_CL_refer, q_CL_transfer_from,
                                          p_MM[0], p_MM_scale_transfer, INT)

        n_stats_H1.append(stats_H1)
        n_stats_H2.append(stats_H2)
        n_stats_L1.append(stats_L1)
        n_stats_L2.append(stats_L2)

        n_MM[i, :, 0] = n_MM1_H + n_MM1_L
        n_MM[i, :, 1] = n_MM2_H + n_MM2_L

        n_nM[i, :, 0] = n_nM1_H + n_nM1_L
        n_nM[i, :, 1] = n_nM2_H + n_nM2_L

    n_stats_H1 = pd.DataFrame(n_stats_H1,
                              columns=['Complication Transfer Rate', 'Initial Complications at Facilities', 'Referrals',
                                       'Initial Complications Post Referral'])
    n_stats_H2 = pd.DataFrame(n_stats_H2,
                              columns=['Complication Transfer Rate', 'Initial Complications at Facilities', 'Referrals',
                                       'Initial Complications Post Referral'])
    n_stats_L1 = pd.DataFrame(n_stats_L1,
                              columns=['Complication Transfer Rate', 'Initial Complications at Facilities', 'Referrals',
                                       'Initial Complications Post Referral'])
    n_stats_L2 = pd.DataFrame(n_stats_L2,
                              columns=['Complication Transfer Rate', 'Initial Complications at Facilities', 'Referrals',
                                       'Initial Complications Post Referral'])

    pph_H = scale_CH_est / (INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b'] * INT['comp']['PP_PPH']['b'] *
                            (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[n_months - 1])
    pph_L = scale_CL_est / (INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b'] *
                            (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[n_months - 1])

    outcomes_dict['Maternal Deaths'] = np.sum(n_MM[i, :, 0])
    outcomes_dict['Neonatal Deaths'] = np.sum(n_nM[i, :, 0])
    outcomes_dict['High PPH'] = (np.sum(n_stats_H1['Initial Complications Post Referral'][n_months - 1]) - np.sum(
        n_MM1_H)) / pph_H
    outcomes_dict['Low PPH'] = (np.sum(n_stats_L1['Initial Complications Post Referral'][n_months - 1]) - np.sum(
        n_MM1_L)) / pph_L

    # still working on
    return n_MM, n_nM, n_stats_H1, n_stats_H2, n_stats_L1, n_stats_L2, outcomes_dict


flags = [flag_sub, flag_CHV, flag_ANC, flag_ANC2, flag_refer, flag_trans, flag_int1, flag_int2, flag_int3, flag_int4]
bn_MM, bn_nM, bn_stats_H1, bn_stats_H2, bn_stats_L1, bn_stats_L2, boutcomes = run_model([1,0,0,0,0,0,0,0,0,0]) # always baseline
n_MM, n_nM, n_stats_H1, n_stats_H2, n_stats_L1, n_stats_L2, outcomes = run_model(flags)                   # intervention, can rename

###############PLOTING HERE####################

categories = list(outcomes.keys())
values1 = list(boutcomes.values())
values2 = list(outcomes.values())

fig, ax = plt.subplots(figsize=(12, 4))
bar_width = 0.35
bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width

# Create side-by-side bar graphs
ax.bar(bar_positions1, values1, width=bar_width, label='Baseline', color='blue')
ax.bar(bar_positions2, values2, width=bar_width, label='With Intervention', color='orange')

# Add labels and title
ax.set_xlabel('Morbidities and Mortalities')
ax.set_ylabel('DALYs')
ax.set_title('DALYs by Severe Health Outcomes')
ax.set_xticks(bar_positions1 + bar_width / 2, categories)  # Set x-axis ticks in the middle of the grouped bars
ax.set_xticklabels(categories)
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

bno_push_L = np.concatenate(bn_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
bno_push_H = np.concatenate(bn_stats_H1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))

bpush_L = np.concatenate(bn_stats_L2['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
bpush_H = np.concatenate(bn_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))

# get complications post referral with intervention
no_push_L = np.concatenate(n_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
no_push_H = np.concatenate(n_stats_H1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
push_L = np.concatenate(n_stats_L2['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
push_H = np.concatenate(n_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))

bpush = bpush_L + bpush_H
push = push_L + push_H


def get_comp_freq(scale_high, scale_low, push0_L, push0_H):
    comp_freq = push0_L / scale_low + push0_H / scale_high
    return comp_freq


def get_ce(boutcomes, outcomes):
    """get cost per DALY averted"""
    cost = 0
    total = 0
    for x, y in zip(boutcomes.values(), outcomes.values()):
        total += (x - y)

    scale_high = (scale_CH_est / (INT['comp']['APH']['inc'] * INT['comp']['Anemia_APH']['b']))
    scale_low = (scale_CH_est / (INT['comp']['APH']['inc'] * INT['comp']['Anemia_APH']['b']))
    scale_high0 = (scale_CH_est / (
                INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b'] * INT['comp']['PP_PPH']['b'] *
                (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[0]))
    scale_high1 = (scale_CH_est / (
                INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b'] * INT['comp']['PP_PPH']['b'] *
                (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[n_months - 1]))
    scale_low0 = (scale_CL_est / (INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b'] *
                                  (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[0]))
    scale_low1 = (scale_CL_est / (INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b'] *
                                  (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[n_months - 1]))

    if flag_int1:
        comps = sum(
            get_comp_freq(scale_high, scale_low, push_L[n_months - 1, :], push_H[n_months - 1, :]) + get_comp_freq(
                scale_high1, scale_low1, push_L[n_months - 1, :], push_H[n_months - 1, :]))
        cost += comps
    if flag_int3:
        cost += 2000 * np.sum(LB_tot) / 100
    if flag_int4:
        cost += 53.68 * LB_tot  # consider, does this actually happen across all facility levels (??) or only the mothers who receive ANC

    # if INT1
    # if INT2
    # if INT3
    # if INT4

    return cost / total

get_ce(boutcomes, outcomes)

scale_high = (scale_CH_est / (INT['comp']['APH']['inc'] * INT['comp']['Anemia_APH']['b']))
scale_low =  (scale_CH_est / (INT['comp']['APH']['inc'] * INT['comp']['Anemia_APH']['b']))
# gets the pushback final values for APH at the first time point
diff_APH = get_comp_freq(scale_high, scale_low, push_L[0,:], push_H[0,:]) - get_comp_freq(scale_high, scale_low, push_L[n_months-1,:], push_H[n_months-1,:])

scale_high0 = (scale_CH_est / (INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b'] *  INT['comp']['PP_PPH']['b'] * (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[0]))
scale_high1 = (scale_CH_est / (INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b'] *  INT['comp']['PP_PPH']['b'] * (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[n_months-1]))
scale_low0 = (scale_CL_est / (INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b']  * (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[0]))
scale_low1 = (scale_CL_est / (INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b']  * (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[n_months-1]))
diff_PPH = get_comp_freq(scale_high0, scale_low0, push_L[0,:], push_H[0,:]) - get_comp_freq(scale_high1, scale_low1, push_L[n_months-1,:], push_H[n_months-1,:])

sum(get_comp_freq(scale_high, scale_low, push_L[n_months-1,:], push_H[n_months-1,:]) + get_comp_freq(scale_high1, scale_low1, push_L[n_months-1,:], push_H[n_months-1,:]))

### referrals related
brno_push = np.concatenate(bn_stats_L1['Referrals'].to_numpy()).reshape((n_months, 4)) + np.concatenate(bn_stats_H1['Referrals'].to_numpy()).reshape((n_months, 4))
brpush = np.concatenate(bn_stats_L2['Referrals'].to_numpy()).reshape((n_months, 4)) + np.concatenate(bn_stats_L1['Referrals'].to_numpy()).reshape((n_months, 4))

rno_push = np.concatenate(n_stats_L1['Referrals'].to_numpy()).reshape((n_months, 4)) + np.concatenate(n_stats_H1['Referrals'].to_numpy()).reshape((n_months, 4))
rpush = np.concatenate(n_stats_L2['Referrals'].to_numpy()).reshape((n_months, 4)) + np.concatenate(n_stats_L1['Referrals'].to_numpy()).reshape((n_months, 4))

to_plotbeg = np.vstack((brpush[0,:], rpush[0,:]))
to_plotend = np.vstack((brpush[n_months-1,:], rpush[n_months-1,:]))

fig, axes = plt.subplots(1,2,figsize = (12,4))
for i in range(2):
    categories = ['Home', 'L2/3', 'L4', 'L5']
    if i==1:
        values1 = to_plotend[0,:]
        values2 = to_plotend[1,:]
    else:
        values1 = to_plotbeg[0,:]
        values2 = to_plotbeg[1,:]

    # Set the width of the bars
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    bar_positions1 = np.arange(len(categories))
    bar_positions2 = bar_positions1 + bar_width

# Create the bar graph
    axes[i].bar(bar_positions1, values1, width=bar_width, label='Group 1')
    axes[i].bar(bar_positions2, values2, width=bar_width, label='Group 2')
    axes[i].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[i].set_xticks(bar_positions1 + bar_width / 2, categories)
    if i ==1:
        axes[i].set_title('Change of LBs due to Referrals (32 months)')
    else:
        axes[i].set_title('Change of LBs due to Referrals (0 months)')
#plt.tight_layout()
plt.legend([' ', 'Baseline', 'With Intervention'])
st.pyplot(fig)

# get complications post referral for baseline
bno_push_L = np.concatenate(bn_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
bno_push_H = np.concatenate(bn_stats_H1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))

bpush_L = np.concatenate(bn_stats_L2['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
bpush_H = np.concatenate(bn_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))

# get complications post referral with intervention
no_push_L = np.concatenate(n_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
no_push_H =  np.concatenate(n_stats_H1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
push_L = np.concatenate(n_stats_L2['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
push_H = np.concatenate(n_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))

### APH Hemorrhage ###
bpush = bpush_L + bpush_H
push = push_L + push_H

aph_effect_beg = (scale_CH_est / (INT['comp']['APH']['inc'] * INT['comp']['Anemia_APH']['b']))
aph_effect_end = (scale_CH_est / (INT['comp']['APH']['inc'] * INT['comp']['Anemia_APH']['b']))
no_effect = (scale_CH_est / INT['comp']['APH']['inc'])

to_plotbeg = np.vstack((push[0,:] / no_effect, push[0,:] / aph_effect_beg))
to_plotend = np.vstack((push[n_months-1,:] / no_effect, push[n_months-1,:] / aph_effect_end))

fig, axes = plt.subplots(1,2,figsize = (12,4))
for i in range(2):
    categories = ['Home', 'L2/3', 'L4', 'L5']
    if i==1:
        values1 = np.clip(list(to_plotend[0,:]), 0, None)
        values2 = np.clip(list(to_plotend[1,:]), 0, None)
        # values1 = to_plotend[0,:]
        # values2 = to_plotend[1,:]
    else:
        values1 = np.clip(list(to_plotbeg[0,:]), 0, None)
        values2 = np.clip(list(to_plotbeg[1,:]), 0, None)
        # values1 = to_plotbeg[0,:]
        # values2 = to_plotbeg[1,:]

    # Set the width of the bars
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    bar_positions1 = np.arange(len(categories))
    bar_positions2 = bar_positions1 + bar_width

# Create the bar graph
    axes[i].bar(bar_positions1, values1, width=bar_width, label='Group 1')
    axes[i].bar(bar_positions2, values2, width=bar_width, label='Group 2')
    axes[i].set_xticks(bar_positions1 + bar_width / 2, categories)
    if i ==1:
        axes[i].set_title('APH Complications by Facility Level (32 months)')
    else:
        axes[i].set_title('APH Complications by Facility Level (0 months)')

#plt.tight_layout()
plt.legend(['Baseline', 'With Intervention'])
st.pyplot(fig)
None

### PPH Hemorrhage ###
pph_effect_beg_H = (scale_CH_est / (INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b'] *  INT['comp']['PP_PPH']['b'] * (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[0]))
pph_effect_end_H = (scale_CH_est / (INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b'] *  INT['comp']['PP_PPH']['b'] * (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[n_months-1]))
pph_effect_beg_L = (scale_CL_est / (INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b']  * (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[0]))
pph_effect_end_L = (scale_CL_est / (INT['comp']['PPH']['inc'] * INT['comp']['Anemia_PPH']['b']  * (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[n_months-1]))

no_effect = (scale_CH_est / INT['comp']['PPH']['inc'])

to_plotbeg = np.vstack((push[0,:] / no_effect, push_L[0,:]/pph_effect_beg_L + push_H[0,:]/pph_effect_beg_H))
to_plotend = np.vstack((push[n_months-1,:] / no_effect, push_L[n_months-1,:]/pph_effect_beg_L + push_H[n_months-1,:]/pph_effect_beg_H))

fig, axes = plt.subplots(1,2,figsize = (12,4))
for i in range(2):
    categories = ['Home', 'L2/3', 'L4', 'L5']
    if i==1:
        values1 = np.clip(list(to_plotend[0,:]), 0, None)
        values2 = np.clip(list(to_plotend[1,:]), 0, None)
        # values1 = to_plotend[0,:]
        # values2 = to_plotend[1,:]
    else:
        values1 = np.clip(list(to_plotbeg[0,:]), 0, None)
        values2 = np.clip(list(to_plotbeg[1,:]), 0, None)
        # values1 = to_plotbeg[0,:]
        # values2 = to_plotbeg[1,:]

    # Set the width of the bars
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    bar_positions1 = np.arange(len(categories))
    bar_positions2 = bar_positions1 + bar_width

# Create the bar graph
    axes[i].bar(bar_positions1, values1, width=bar_width, label='Group 1')
    axes[i].bar(bar_positions2, values2, width=bar_width, label='Group 2')
    axes[i].set_xticks(bar_positions1 + bar_width / 2, categories)
    if i ==1:
        axes[i].set_title('PPH Complications by Facility Level (32 months)')
    else:
        axes[i].set_title('PPH Complications by Facility Level (0 months)')

#plt.tight_layout()
plt.legend(['Baseline', 'With Intervention'])
st.pyplot(fig)
None

no_push = np.concatenate(n_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4)) + np.concatenate(n_stats_H1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
push = np.concatenate(n_stats_L2['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4)) + np.concatenate(n_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
fig, (ax1, ax2)= plt.subplots(1,2, figsize=(12, 4))
for j in range(4):
        ax1.plot(range(n_months), no_push)
        ax2.plot(range(n_months), push)

plt.suptitle('Complications facility level - Pushback')
ax1.set_xlabel('Time')
ax2.set_xlabel('Time')
ax1.set_ylabel('Complications')
ax2.set_ylabel('Complications')
#plt.tight_layout()
#plt.show()
st.pyplot(fig)

# exclude home births

num_rows, num_cols = 1, 2

# Create a grid of subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))

# Iterate over j from 0 to 1 (two subplots)
for j in range(2):
    # Replace 0 with j to access different data
    nM = n_nM[:, :, j].T

    # Plot the data in the current subplot
    for i in range(1, 4):
        axes[j].plot(range(n_months), nM[i, :])
    axes[j].set_title(f'Pushback {j}')
    axes[j].set_xlabel('Months')
    axes[j].set_ylabel('Deaths')

plt.suptitle('Neonatal deaths by facility level - Pushback')
#plt.tight_layout()
#plt.show()
st.pyplot(fig)

fig, axes = plt.subplots(4, 3, figsize=(12, 8))

# Loop through the subplots and plot the data
j=0
for x in range(4):
    for y in range(3):
        ax = axes[x, y]
        for i in range(4):
            ax.plot(range(n_months-1), n_MMs[1:, i, j, 0])
            ax.set_title(f'SC {j}')
            ax.set_xlabel('Months')
            ax.set_ylabel('Deaths')
        j+=1

plt.suptitle('Maternal deaths by subcounty and facility level')
#plt.tight_layout()
#plt.show()
st.pyplot(fig)

num_rows, num_cols = 1, 2

# Create a grid of subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))

# Iterate over j from 0 to 1 (two subplots)
for j in range(2):
    # Replace 0 with j to access different data
    MM = n_MM[:, :, j].T

    # Plot the data in the current subplot
    for i in range(4):
        axes[j].plot(range(n_months), MM[i, :])
    axes[j].set_title(f'Pushback {j}')
    axes[j].set_xlabel('Months')
    axes[j].set_ylabel('Deaths')

plt.suptitle('Maternal deaths by facility level - Pushback')
#plt.tight_layout()
#plt.show()
st.pyplot(fig)

num_rows, num_cols = 4, 3

# Create a grid of subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

# Iterate over j from 0 to 11
for j in range(num_rows * num_cols):
    # Replace 7 with j to access different LB1s
    y_values = SC['LB1s'][j]
    y_values = y_values.T

    # Calculate the row and column indices for the current subplot
    row, col = divmod(j, num_cols)

    # Plot the data in the current subplot
    for i in range(4):
        axes[row, col].plot(range(n_months), y_values[i])

    axes[row, col].set_title(f'Subcounty {j + 1}')
    axes[row, col].set_xlabel('Months')
    axes[row, col].set_ylabel('# of Births')

fig.suptitle('Live births by subcounty and facility level')
#plt.tight_layout()
#plt.show()
st.pyplot(fig)

##############PLOT in Streamlit version##################
# if flag_sub:
#     num_rows, num_cols = 4, 3
#     plot_columns = [st.columns(num_cols) for _ in range(num_rows)]
#
#     for j in range(12):
#         facility_levels = [f'level{i}' for i in range(SC['LB1s'][j].T.shape[0])]
#         months = [i + 1 for i in range(SC['LB1s'][j].T.shape[1])]
#
#         data = {
#             "level": [level for level in facility_levels for _ in range(SC['LB1s'][j].T.shape[1])],
#             "month": [month for _ in range(SC['LB1s'][j].T.shape[0]) for month in months],
#             "value": SC['LB1s'][j].T.flatten()  # You can name this column as needed
#         }
#
#         df = pd.DataFrame(data)
#
#         chart = (
#             alt.Chart(
#                 data=df,
#                 title=f'Subcounty {j + 1}',
#             )
#             .mark_line()
#             .encode(
#                 x=alt.X("month", axis=alt.Axis(title="Month")),
#                 y=alt.Y("value", axis=alt.Axis(title="# of Births")).scale(domain=(0, 4500)),
#                 color=alt.Color("level:N").title("Level"),
#             ).properties(
#             width=300,
#             height=250
#             )
#         )
#
#         chart = chart.properties(
#         ).configure_title(
#         anchor='middle'
#         )
#
#         row = j // num_cols
#         col = j % num_cols
#
#         with plot_columns[row][col]:
#             st.altair_chart(chart)
#
# ##Plot death at Kakamega level
# dfs = []
# for j in range(2):
#     facility_levels = [f'{i}' for i in range(n_MM[:, :, j].T.shape[0])]
#     months = [i + 1 for i in range(n_MM[:, :, j].T.shape[1])]
#
#     data = {
#         "level": [level for level in facility_levels for _ in range(n_MM[:, :, j].T.shape[1])],
#         "month": [month for _ in range(n_MM[:, :, j].T.shape[0]) for month in months],
#         "value": n_MM[:, :, j].T.flatten()  # You can name this column as needed
#     }
#
#     data["pushback"] = j
#     df = pd.DataFrame(data)
#     dfs.append(df)
#
# dfs = pd.concat(dfs, ignore_index=True)
# dfs['level'] = dfs['level'].astype('category')
#
# # Filter the DataFrame to create subsets for pushback = 0 and pushback = 1
# df_pushback_0 = dfs[dfs['pushback'] == 0]
# df_pushback_0 = df_pushback_0.loc[df_pushback_0['level'].isin(['1', '2', '3']),:]
# df_pushback_1 = dfs[dfs['pushback'] == 1]
# df_pushback_1 = df_pushback_1.loc[df_pushback_1['level'].isin(['1', '2', '3']),:]
#
# col1, col2 = st.columns(2)
# with col1:
#     chart_pushback_0 = (
#         alt.Chart(
#             data=df_pushback_0,
#             title="Pushback 0",
#         )
#         .mark_line()
#         .encode(
#             x=alt.X("month", axis=alt.Axis(title="Month")),
#             y=alt.Y("value", axis=alt.Axis(title="Deaths")).scale(domain=(0, 50)),
#             color=alt.Color("level:N").title("Level")
#         )
#     )
#
#     chart_pushback_0 = chart_pushback_0.properties(
#     ).configure_title(
#     anchor='middle'
#     )
#
#     st.altair_chart(chart_pushback_0)
#
# with col2:
#     chart_pushback_1 = (
#         alt.Chart(
#             data=df_pushback_1,
#             title="Pushback 1",
#         )
#         .mark_line()
#         .encode(
#             x=alt.X("month", axis=alt.Axis(title="Month")),
#             y=alt.Y("value", axis=alt.Axis(title="Deaths")).scale(domain=(0, 50)),
#             color=alt.Color("level:N").title("Level")
#         )
#     )
#     chart_pushback_1 = chart_pushback_1.properties(
#     ).configure_title(
#     anchor='middle'
#     )
#
#     st.altair_chart(chart_pushback_1)
#
# ##Plot live bith by subcounty level
# num_rows, num_cols = 4, 3
# plot_columns = [st.columns(num_cols) for _ in range(num_rows)]
#
# for j in range(12):
#     facility_levels = [f'level{i}' for i in range(SC['LB1s'][j].T.shape[0])]
#     months = [i + 1 for i in range(SC['LB1s'][j].T.shape[1])]
#
#     data = {
#         "level": [level for level in facility_levels for _ in range(SC['LB1s'][j].T.shape[1])],
#         "month": [month for _ in range(SC['LB1s'][j].T.shape[0]) for month in months],
#         "value": SC['LB1s'][j].T.flatten()  # You can name this column as needed
#     }
#
#     df = pd.DataFrame(data)
#
#     chart = (
#         alt.Chart(
#             data=df,
#             title=f'Subcounty {j + 1}',
#         )
#         .mark_line()
#         .encode(
#             x=alt.X("month", axis=alt.Axis(title="Month")),
#             y=alt.Y("value", axis=alt.Axis(title="# of Births")).scale(domain=(0, 4500)),
#             color=alt.Color("level:N").title("Level"),
#         ).properties(
#         width=300,
#         height=250
#         )
#     )
#
#     chart = chart.properties(
#     ).configure_title(
#     anchor='middle'
#     )
#
#     row = j // num_cols
#     col = j % num_cols
#
#     with plot_columns[row][col]:
#         st.altair_chart(chart)
