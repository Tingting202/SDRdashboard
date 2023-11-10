import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
from sympy import symbols, Eq, solve
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

#Not all subcounty has the capacility to deal with C-sections, they need to refer to L5 in subcounty 6.
C_section_subcounty = [
    [0, 0, 259, None],
    [0,	0, 0, None],
    [0,	0, 245, None],
    [0,	0, 95,	None],
    [0,	0, 159,	None],
    [0,	222, None, 1526],
    [6,	0, 307, None],
    [0,	0, 0, None],
    [0,	0, 0, None],
    [0, 0, 950, None],
    [0, 0, 87, None],
    [0, 0, 382, None]
]

# n_CH.refer
n_CH_refer = np.array([
    [0, -93, 80, 42]
])

# n_CL.refer
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
        0: np.zeros((n_months,4)),
        1: np.zeros((n_months,4)),
        2: np.zeros((n_months,4)),
        3: np.zeros((n_months,4)),
        4: np.zeros((n_months,4)),
        5: np.zeros((n_months,4)),
        6: np.zeros((n_months,4)),
        7: np.zeros((n_months,4)),
        8: np.zeros((n_months,4)),
        9: np.zeros((n_months,4)),
        10: np.zeros((n_months,4)),
        11: np.zeros((n_months,4)),
    },
    'LB2s': {
        0: np.zeros((n_months,4)),
        1: np.zeros((n_months,4)),
        2: np.zeros((n_months,4)),
        3: np.zeros((n_months,4)),
        4: np.zeros((n_months,4)),
        5: np.zeros((n_months,4)),
        6: np.zeros((n_months,4)),
        7: np.zeros((n_months,4)),
        8: np.zeros((n_months,4)),
        9: np.zeros((n_months,4)),
        10: np.zeros((n_months,4)),
        11: np.zeros((n_months,4)),
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
    SC['LB1s'][i][0,:] = LB1s[i]
    SC['LB2s'][i][0,:] = LB1s[i]

for i in range(SC['n']):
    n_CHs['refer'][i] = CHs[i]
    n_CLs['refer'][i] = CLs[i]

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
    subint = st.selectbox('Subcounty outcome', list(options.keys()))
    flag_sub = options[subint]
    ANCint = st.selectbox('ANC intervention', list(options.keys()))
    flag_ANC= options[ANCint]
    ANC2int = st.selectbox('ANC2 intervention', list(options.keys()))
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
    int7 = st.selectbox('Ultrasound', list(options.keys()))
    flag_int7 = options[int7]
    diagnosisrate = st.slider('Ultrasound diagnosis rate', min_value=0.0, max_value=1.0, step=0.1, value=0.5)

INT = {
    'CHV': {
        'n_months_push_back': n_months_push_back,
        'b_push_back': b_push_back,
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
            'b': 0.5*0.017*(1-0.812)/scale_CH_est
        },
        'ANC_msepsis': {
            'b': None
        },
        'ANC_eclampsia': {
            'b': None
        },
        'ANC_obs_labor': {
            'b': 0.5*0.1*(1-0.62)/scale_CH_est
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
        'refer_rate': 0
    },
    'trans': {
        'b0': 0,
        'bs': np.ones(n_months),
        'b': None
    },
    'comp': {
        'PPH': {
            'inc': 0.017*0.5,
            'b0': 1,
            'b': 1
        },
        'APH': {
            'inc': 0.0039*0.5,
            'b0': 1,
            'b': 1
        },
        'msepsis': {
            'inc': 0.00159*0.5
        },
        'eclampsia': {
            'inc': 0.0046*0.5
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
            'inc': 0.0027*0.5,
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
    'ultrasound':{
        'diagnosis_rate': 0
    }
}

Capacity_ratio1 = np.zeros((n_months,SC['n']))
Capacity_ratio2 = np.zeros((n_months,SC['n']))
Push_back = np.zeros((n_months,SC['n']))
n_MM = np.zeros((n_months,4,2)) # number of months, for each hospital level, with and without pushback
n_nM = np.zeros((n_months,4,2)) # neonatal mortality
n_MMs = np.zeros((n_months,4,12,2))
p_MMs = np.zeros((n_months,4,12,2))

####Referral due to Ultrasound at whole county level#######
#Calculate the number of mothers with placenta previa and placental abruption
N_pp = np.array(LB_tot) * 0.0027  #placenta previa
N_pa = np.array(LB_tot) * 0.01  #placental abruption
N_pph_bypp = N_pp * 0.0199
N_pph_bypa = N_pa * 0.182
N_preterm_bypp = N_pp * 0.559
N_preterm_bypa = N_pa * 0.396
N_preterm_byother = np.array(LB_tot) * 0.023523 - (N_preterm_bypp + N_preterm_bypa)
#Calculate the number of PPH, APH, and preterm from our facility data
N_pph_facility = np.array(LB_tot) * 0.017
N_preterm_facility = np.array(LB_tot) * 0.023523

#Calculate the number of pph, aph, and preterm can be detected due to ultrasound diagnosis
N_pph_detected = N_pph_bypp * 0.875 + N_pph_bypa * 0.57
N_preterm_detected = N_preterm_bypp * 0.875 + N_preterm_bypa * 0.57 + N_preterm_byother * 0.99
#Assume all detected pph in L2/L3 will be referred to L4 or L5
N_pph_refered = np.array([0, -N_pph_detected[1], N_pph_detected[1]*0.5, N_pph_detected[1]*0.5])
N_preterm_refered = np.array([0, -N_preterm_detected[1], N_preterm_detected[1]*0.5, N_preterm_detected[1]*0.5])

### CHOOSE INTERVENTIONS ###

def comp_reduction(oddsratio, p_expose, p_comp, int_expose):
    """given odds ratio and exposure, return change in complications due to change in exposure"""

    x, y = symbols('x y')
    eq1 = Eq(x/(1-x)/(y/(1-y))-oddsratio,0)
    eq2 = Eq(p_comp - p_expose*x - (1-p_expose)*y,0)
    solution = solve((eq1,eq2), (x, y))[0]

    base_comp = solution[0]*p_expose + solution[1]*(1-p_expose)
    new_comp = solution[0]*p_expose*int_expose + solution[1]*(1-p_expose*int_expose)

    change = (scale_CH_est - base_comp + new_comp)/scale_CH_est
    return change

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

if flag_trans:
    INT['trans']['b0'] = 1
    INT['trans']['bs'] = INT['trans']['b0'] * (t < n_months / 2)
else:
    INT['trans']['bs'] = np.ones(n_months)
if flag_int1:
    INT['comp']['PPH']['b'] = 1 - 0.6 * INT['comp']['PPH']['inc']/(scale_CH_est) # difference from original
else:
    INT['comp']['PPH']['b'] = 1
if flag_int2:
    INT['comp']['Anemia_PPH']['b'] = comp_reduction(3.54, INT['comp']['Anemia']['inc'], INT['comp']['PPH']['inc'], 0.3)
    INT['comp']['Anemia_APH']['b'] = comp_reduction(1.522, INT['comp']['Anemia']['inc'], INT['comp']['APH']['inc'], 0.3)
else:
    INT['comp']['Anemia_PPH']['b'] = 1
    INT['comp']['Anemia_APH']['b'] = 1
if flag_int4:
    if flag_ANC:
        INT['mort']['red'] = 0.038
else:
    INT['mort']['red'] = 0
if flag_int7:
    INT['ultrasound']['diagnosis_rate']= diagnosisrate
else:
    INT['ultrasound']['diagnosis_rate'] = 0

INT['ANC']['bs'] = 1 - (1 - INT['ANC']['b0']) ** np.minimum(t, 12)
INT['refer']['bs'] = INT['refer']['b0'] * (t - 1)

INT['ANC']['ANC_msepsis']['b'] = np.array([comp_reduction(4, INT['comp']['msepsis']['inc'], INT['ANC']['inc'], i) for i in (1-INT['ANC']['bs']/(INT['ANC']['b0']*12))])
INT['ANC']['ANC_eclampsia']['b'] = np.array([comp_reduction(1.41, INT['comp']['eclampsia']['inc'], INT['ANC']['inc'], i) for i in (1-INT['ANC']['bs']/(INT['ANC']['b0']*12))])
INT['ANC']['ANC_rup_uterus']['b'] = np.array([comp_reduction(6.29, INT['comp']['rup_uterus']['inc'], INT['ANC']['inc'], i) for i in (1-INT['ANC']['bs']/(INT['ANC']['b0']*12))])

if flag_ANC2:
    flag_ANC = 1
    INT['ANC']['bs'] = INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b'] * INT['ANC']['ANC_msepsis']['b'] * INT['ANC']['ANC_eclampsia']['b'] * INT['ANC']['ANC_obs_labor']['b'] * INT['ANC']['ANC_rup_uterus']['b'])
    INT['mort']['b'] = INT['ANC']['ANC_preterm']['b']


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
    if np.sum(q_C0_transfer_from)>0:
        n_initial_est_k = LB_tot_k * scale_CH_est * INT['comp']['Anemia_PPH']['b'] * INT['comp']['Anemia_APH']['b'] * INT['comp']['PP_PPH']['b']# number of complications original
    else:
        n_initial_est_k = LB_tot_k * scale_CH_est * INT['comp']['Anemia_PPH']['b'] * INT['comp']['Anemia_APH']['b']
    n_refer_k = n_refer_k * (1+INT['refer']['b'])                    # change in referral due to intervention
    n_initial_C = n_refer_k + n_initial_est_k                        # initial complications
    n_initial_C = n_initial_C.flatten()
    n_transfer_from_k = np.diag(n_initial_C[0:3]) @ q_C0_transfer_from       # number transferred given original complications and transfer probabilities
    n_transfer_out = np.append(np.sum(n_transfer_from_k, axis=1), 0)         # transferred out
    n_transfer_in = np.sum(n_transfer_from_k, axis=0)                        # transferred in
    n_not_transfer = n_initial_C - n_transfer_out                            # not transferred
    n_MM = n_MM + np.maximum(0, n_not_transfer) * p_MM                       # considering transfer, not transfer, effect on mortality
    n_MM = n_MM + n_transfer_in * p_MM * p_MM_scale_transfer

    n_nM = n_MM * scale_N
    # n_nM = max(0, n_nM - INT['mort']['red'] * LB_tot_k)

    return n_MM, n_nM

for i in t:
    LB1_tot_i = np.zeros(4)
    LB2_tot_i = np.zeros(4)
    INT['refer']['b'] = INT['refer']['bs'][i]
    INT['ANC']['b'] = INT['ANC']['bs'][i]
    INT['trans']['b'] = INT['trans']['bs'][i]
    scale_CH_est_i = scale_CH_est * (1 - INT['ANC']['b'])
    scale_CL_est_i = scale_CL_est * (1 - INT['ANC']['b'])

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

                n_MM1s_H, _ = f_MM(LB_tot_j, SDR_multiplier_1_j, scale_CH_est_i, n_CHs['refer'][j], q_CH_transfer_from,
                                   p_MM[1], p_MM_scale_transfer, INT)
                n_MM2s_H, _ = f_MM(LB_tot_j, SDR_multiplier_2_j, scale_CH_est_i, n_CHs['refer'][j], q_CH_transfer_from,
                                   p_MM[1], p_MM_scale_transfer, INT)
                n_MM1s_L, _ = f_MM(LB_tot_j, SDR_multiplier_1_j, scale_CL_est_i, n_CLs['refer'][j], q_CL_transfer_from,
                                   p_MM[0], p_MM_scale_transfer, INT)
                n_MM2s_L, _ = f_MM(LB_tot_j, SDR_multiplier_2_j, scale_CL_est_i, n_CLs['refer'][j], q_CL_transfer_from,
                                   p_MM[0], p_MM_scale_transfer, INT)

                n_MMs[i, :, j, 0] = n_MM1s_H + n_MM1s_L
                n_MMs[i, :, j, 1] = n_MM2s_H + n_MM2s_L
                p_MMs[i, :, j, :] = n_MMs[i, :, j, :] / LB_tot_j[:, None]

        Capacity_ratio1[i, j] = np.sum(SC['LB1s'][j][i, 2:4]) / Capacity[j]  # LB/capacity (no pushback)
        Capacity_ratio2[i, j] = np.sum(SC['LB2s'][j][i, 2:4]) / Capacity[j]  # (pushback)
        LB1_tot_i += SC['LB1s'][j][i, :]
        LB2_tot_i += SC['LB2s'][j][i, :]

    SDR_multiplier_1 = LB1_tot_i / LB_tot
    SDR_multiplier_2 = LB2_tot_i / LB_tot
    n_MM1_H, n_nM1_H = f_MM(LB_tot, SDR_multiplier_1, scale_CH_est_i, n_CH_refer, q_CH_transfer_from, p_MM[1],
                            p_MM_scale_transfer, INT)
    n_MM2_H, n_nM2_H = f_MM(LB_tot, SDR_multiplier_2, scale_CH_est_i, n_CH_refer, q_CH_transfer_from, p_MM[1],
                            p_MM_scale_transfer, INT)
    n_MM1_L, n_nM1_L = f_MM(LB_tot, SDR_multiplier_1, scale_CL_est_i, n_CL_refer, q_CL_transfer_from, p_MM[0],
                            p_MM_scale_transfer, INT)
    n_MM2_L, n_nM2_L = f_MM(LB_tot, SDR_multiplier_2, scale_CL_est_i, n_CL_refer, q_CL_transfer_from, p_MM[0],
                            p_MM_scale_transfer, INT)

    n_MM[i, :, 0] = n_MM1_H + n_MM1_L
    n_MM[i, :, 1] = n_MM2_H + n_MM2_L

    n_nM[i, :, 0] = n_nM1_H + n_nM1_L
    n_nM[i, :, 1] = n_nM2_H + n_nM2_L

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
df_pushback_0 = df_pushback_0.loc[df_pushback_0['level'].isin(['1', '2', '3']),:]
df_pushback_1 = dfs[dfs['pushback'] == 1]
df_pushback_1 = df_pushback_1.loc[df_pushback_1['level'].isin(['1', '2', '3']),:]

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
            y=alt.Y("value", axis=alt.Axis(title="Deaths")).scale(domain=(0, 50)),
            color=alt.Color("level:N").title("Level")
        )
    )

    chart_pushback_0 = chart_pushback_0.properties(
    ).configure_title(
    anchor='middle'
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
            y=alt.Y("value", axis=alt.Axis(title="Deaths")).scale(domain=(0, 50)),
            color=alt.Color("level:N").title("Level")
        )
    )
    chart_pushback_1 = chart_pushback_1.properties(
    ).configure_title(
    anchor='middle'
    )

    st.altair_chart(chart_pushback_1)


num_rows, num_cols = 4, 3
plot_columns = [st.columns(num_cols) for _ in range(num_rows)]

for j in range(12):
    facility_levels = [f'level{i}' for i in range(SC['LB1s'][j].T.shape[0])]
    months = [i + 1 for i in range(SC['LB1s'][j].T.shape[1])]

    data = {
        "level": [level for level in facility_levels for _ in range(SC['LB1s'][j].T.shape[1])],
        "month": [month for _ in range(SC['LB1s'][j].T.shape[0]) for month in months],
        "value": SC['LB1s'][j].T.flatten()  # You can name this column as needed
    }

    df = pd.DataFrame(data)

    chart = (
        alt.Chart(
            data=df,
            title=f'Subcounty {j + 1}',
        )
        .mark_line()
        .encode(
            x=alt.X("month", axis=alt.Axis(title="Month")),
            y=alt.Y("value", axis=alt.Axis(title="# of Births")).scale(domain=(0, 4500)),
            color=alt.Color("level:N").title("Level"),
        ).properties(
        width=300,
        height=250
        )
    )

    chart = chart.properties(
    ).configure_title(
    anchor='middle'
    )

    row = j // num_cols
    col = j % num_cols

    with plot_columns[row][col]:
        st.altair_chart(chart)
