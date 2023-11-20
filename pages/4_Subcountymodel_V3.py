import streamlit as st
from matplotlib import pyplot as plt
import geopandas as gpd
import numpy as np
from sympy import symbols, Eq, solve
import plotly.express as px
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")
st.subheader("Choose interventions")
#st.set_page_config(layout="wide")
options = {
    'Off': 0,
    'On': 1
}

selected_plotA = None
selected_plotB = None
selected_plotC = None
flag_sub = 0

### Side bar ###
with st.sidebar:
    st.header("Outcomes sidebar")
    level_options = ("County", "Subcounty", "Subcounty in Map")
    select_level = st.selectbox('Select level of interest:', level_options)
    if select_level == "County":
        plotA_options = ("Live births", "Neonatal deaths", "Maternal deaths", "Cost effectiveness", "DALYs", "PPH complications") #To be added: "MMR", "Effects of CHV pushback"
        selected_plotA = st.selectbox(
            label="Select outcome of interest:",
            options=plotA_options,
        )
    if select_level == "Subcounty":
        flag_sub = 1
        plotB_options = ("Maternal deaths", "Neonatal deaths", "Live births")
        selected_plotB = st.selectbox(
            label="Select outcome of interest:",
            options=plotB_options,
        )
    if select_level == "Subcounty in Map":
        flag_sub = 1
        plotC_options = ("MMR", "NMR", "% deliveries in L4/5")
        selected_plotC = st.selectbox(
            label="Choose your interested outcomes in map:",
            options=plotC_options,
        )
### Sliders ###
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.text("Healthcare")
    ANCint = st.checkbox('ANC')
    if ANCint:
        flag_ANC = 1
        ANC2int = st.selectbox('ANC effect on complications', list(options.keys()))
        flag_ANC2 = options[ANC2int]
    else:
        flag_ANC = 0
        flag_ANC2 = 0

    Referint = st.checkbox('Referal')
    if Referint:
        flag_refer = 1
        referrate = st.slider('refer rate', min_value=0.0, max_value=1.0, step=0.1, value=0.5)
    else:
        flag_refer = 0
        referrate = 0
with col2:
    st.text("Healthcare")
    int1 = st.checkbox('Obstetric drape')
    if int1:
        flag_int1 = 1
    else:
        flag_int1 = 0

    int2 = st.checkbox('IV Iron')
    if int2:
        flag_int2 = 1
    else:
        flag_int2 = 0

    int4 = st.checkbox('Antenatal corticosteroids')
    if int4:
        flag_int4 = 1
    else:
        flag_int4 = 0

    int3 = st.checkbox('Ultrasound')
    if int3:
        flag_int3 = 1
        diagnosisrate = st.slider('Ultrasound diagnosis rate', min_value=0.0, max_value=1.0, step=0.1, value=0.5)
    else:
        flag_int3 = 0
        diagnosisrate = 0
with col3:
    st.text("Infrastructure")
    CHVint = st.checkbox('CHV')
    if CHVint:
        flag_CHV = 1
    else:
        flag_CHV = 0
    Transint = st.checkbox('Increase transfer by 25%')
    if Transint:
        flag_trans = 1
    else:
        flag_trans = 0
    Transint2 = st.checkbox('Stop transfer after the halfway point')
    if Transint2:
        flag_trans2 = 1
    else:
        flag_trans2 = 0
with col4:
    st.text("SDR")


### Data files ###
# df = pd.read_excel('DHS_Cluster.xlsx', sheet_name='DHS 2022')
# df_subcounty = pd.read_excel('DHS_Cluster.xlsx', sheet_name='Cluster_Subcounty_2022')
# df = df.merge(df_subcounty, left_on='cluster', right_on='clusterID')
# anc_dist = df[['Subcounty', 'anc', 'delivery']].groupby(['Subcounty', 'delivery']).value_counts(normalize=True).reset_index()
shapefile_path2 = 'ke_subcounty.shp'


### PARAMETERs ###
subcounties = [
    "Butere",
    "Ikolomani",
    "Khwisero",
    "Likuyani",
    "Lugari",
    "Lurambi",
    "Malava",
    "Matungu",
    "Mumias East",
    "Mumias West",
    "Navakholo",
    "Shinyalu"
]

comps = ['Low PPH', 'High PPH', 'Neonatal Deaths', 'Maternal Deaths']
DALYs = [0.114, 0.324, 1, 0.54]
DALY_dict = {x: 0 for x in comps}

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
scale_N = 16                # neonatal mortality

### MODEL PERTINENT ###
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
    },
    'Class': {
        0: 0.767669453,
        1: 0.707382485,
        2: 0.583722902,
        3: 0.353483752,
        4: 0.621879554,
        5: 0.341297702,
        6: 0.78264497,
        7: 0.881471916,
        8: 0.815330894,
        9: 0.583155486,
        10: 0.759252685,
        11: 0.708221685,
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

### ATTEMPT TO STANDARDIZE FORMAT BELOW ###
# ex: ANC_eclampsia -> effect of ANC on reduction in eclampsia (direct reduction in complications)
# complication reductions will go

# FIRST LEVEL COMPLICATION INTERVENTION: intervention directly reduces initial number of complications
# SECOND LEVEL COMPLICATION INTERVENTION: intervention reduces some other effect, for example, ANC, that then reduces complications
# FIRST LEVEL MORTALITY INTERVENTION: intervention directly affects mortality due to a specific cause

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
        'refer_rate': 0 # 1 #0
    },
    'trans': {
        'b0': 0,
        'bs': np.zeros(n_months),
        'b': None
    },
    'mcomp': {
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
    'ncomp': {
        'preterm': {
            'inc': 0.002438884431,
            'b': 1,                   # this actually reduces the number of neonatal mortalities
            'int_effect': 0.84       # intervention effect
        }
    },
    'mort': {
        'red': 0,
        'b': 0,
        'n_preterm': 0.1036808413,
        'n_overall': 0.013699,
        'n_asphyxia': 0.004313/0.013699,
        'n_sepsis': 0.002237/0.013699
    },
    'ultrasound':{
        'diagnosis_rate': 0 # 1 #0
    }
}
#INT['mort']['n_asphyxia']

Capacity_ratio1 = np.zeros((n_months,SC['n']))
Capacity_ratio2 = np.zeros((n_months,SC['n']))
Push_back = np.zeros((n_months,SC['n']))
n_MM = np.zeros((n_months,4,2)) # number of months, for each hospital level, with and without pushback
n_nM = np.zeros((n_months,4,2)) # neonatal mortality
n_MMs = np.zeros((n_months,4,12,2))
p_MMs = np.zeros((n_months,4,12,2))
n_nMs = np.zeros((n_months,4,12,2))
p_nMs = np.zeros((n_months,4,12,2))

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

def set_flags(flags):
    flag_sub = flags[0]
    flag_CHV = flags[1]
    flag_ANC = flags[2]
    flag_ANC2 = flags[3]
    flag_refer = flags[4]
    # flag_trans = flags[5]
    flag_trans = flags[5]
    flag_int1 = flags[6]   # Obstetric drape
    # this would be implemented everywhere except home
    flag_int2 = flags[7]   # Anemia reduction through IV Iron
    # this occurs through ANC, so should be okay everywhere (we should consider ANC incidence - above baseline)
    flag_int3 = flags[8]   # ultrasound
    # this occurs through ANC, so should be okay everywhere (we should consider ANC incidence - above baseline)
    flag_int4 = flags[9]   # antenatal corticosteroids - reduction of neonatal mortality
    # this occurs through ANC, so should also be okay everywhere
    flag_trans2 = flags[10]

    # Update dictionary values based on flags
    if flag_CHV:
        INT['CHV']['L4']['b'] = 0.02
    else:
        INT['CHV']['L4']['b'] = 0
    if flag_ANC:
        INT['ANC']['b0'] = 0.03
    else:
        INT['ANC']['b0'] = 0
    if flag_refer:
        INT['refer']['b0'] = 0.1
    else:
        INT['refer']['b0'] = 0
    if flag_trans:
        #"""transfer for class 2 bumps up from 1 to 1.25"""
        INT['trans']['bs'] = np.repeat(1.25,n_months)
    else:
        INT['trans']['bs'] = np.ones(n_months)
    if flag_trans2:
        #"""transfer for class 2 stays at 1, but stops after the halfway point"""
        INT['trans']['b0'] = np.ones(n_months)
        INT['trans']['bs'] = INT['trans']['b0'] * (t < n_months / 2)
    else:
        INT['trans']['bs'] = np.ones(n_months)
    if flag_int1:
        INT['mcomp']['PPH']['b'] = np.array([1, 1 - 0.6 * INT['mcomp']['PPH']['inc']/(scale_CH_est), 1 - 0.6 * INT['mcomp']['PPH']['inc']/(scale_CH_est), 1 - 0.6 * INT['mcomp']['PPH']['inc']/(scale_CH_est)]) # difference from original
    else:
        INT['mcomp']['PPH']['b'] = 1
    if flag_int2:
        INT['mcomp']['Anemia_PPH']['b'] = comp_reduction(3.54, INT['mcomp']['Anemia']['inc'], INT['mcomp']['PPH']['inc'], 0.3)
        INT['mcomp']['Anemia_APH']['b'] = comp_reduction(1.522, INT['mcomp']['Anemia']['inc'], INT['mcomp']['APH']['inc'], 0.3)
    else:
        INT['mcomp']['Anemia_PPH']['b'] = 1
        INT['mcomp']['Anemia_APH']['b'] = 1
    if flag_int3:
        INT['ultrasound']['diagnosis_rate'] = diagnosisrate
    else:
        INT['ultrasound']['diagnosis_rate'] = 0
    if flag_int4:
        if flag_ANC:
            new_mort = INT['ncomp']['preterm']['inc']*INT['mort']['n_preterm']*INT['ncomp']['preterm']['int_effect']
            INT['ncomp']['preterm']['b'] = (INT['mort']['n_overall'] - new_mort)/INT['mort']['n_overall']
    else:
        INT['ncomp']['preterm']['b'] = 1

    INT['ANC']['bs'] = 1 - (1 - INT['ANC']['b0']) ** np.minimum(t, 12)
    INT['refer']['bs'] = INT['refer']['b0'] * (t - 1)

    if flag_ANC:
        #"""effect of ANC on complication reduction for regular ANC"""
        INT['ANC']['ANC_msepsis']['b'] = np.array([comp_reduction(4, INT['mcomp']['msepsis']['inc'], INT['ANC']['inc'], i) for i in (1-INT['ANC']['bs']/(INT['ANC']['b0']*12))])
        INT['ANC']['ANC_eclampsia']['b'] = np.array([comp_reduction(1.41, INT['mcomp']['eclampsia']['inc'], INT['ANC']['inc'], i) for i in (1-INT['ANC']['bs']/(INT['ANC']['b0']*12))])
        INT['ANC']['ANC_rup_uterus']['b'] = np.array([comp_reduction(6.29, INT['mcomp']['rup_uterus']['inc'], INT['ANC']['inc'], i) for i in (1-INT['ANC']['bs']/(INT['ANC']['b0']*12))])

    if flag_ANC2:
        #"""effect of ANC on complication reduction on top of regular ANC"""
        flag_ANC = 1
        INT['ANC']['bs'] = INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b'] * INT['ANC']['ANC_msepsis']['b'] * INT['ANC']['ANC_eclampsia']['b'] * INT['ANC']['ANC_obs_labor']['b'] * INT['ANC']['ANC_rup_uterus']['b'])
        INT['mort']['b'] = INT['ANC']['ANC_preterm']['b']

    return INT

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

    n_nM = n_MM * scale_N * INT['ncomp']['preterm']['b']
    # n_nM = max(0, n_nM - INT['mort']['red'] * LB_tot_k)

    stats = [q_C0_transfer_from, n_initial_est_k, n_refer_k, n_initial_C]

    return n_MM, n_nM, stats


def run_model(flags):
    """runs whole model"""

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
        },
        'Class': {
            0: 0.767669453,
            1: 0.707382485,
            2: 0.583722902,
            3: 0.353483752,
            4: 0.621879554,
            5: 0.341297702,
            6: 0.78264497,
            7: 0.881471916,
            8: 0.815330894,
            9: 0.583155486,
            10: 0.759252685,
            11: 0.708221685,
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
    Capacity_ratio1 = np.zeros((n_months, SC['n']))
    Capacity_ratio2 = np.zeros((n_months, SC['n']))
    Push_back = np.zeros((n_months, SC['n']))
    n_MM = np.zeros((n_months, 4, 2))  # number of months, for each hospital level, with and without pushback
    n_nM = np.zeros((n_months, 4, 2))  # neonatal mortality
    n_MMs = np.zeros((n_months, 4, 12, 2))
    p_MMs = np.zeros((n_months, 4, 12, 2))
    n_nMs = np.zeros((n_months, 4, 12, 2))
    p_nMs = np.zeros((n_months, 4, 12, 2))
    outcomes_dict = {x: 0 for x in comps}

    INT = set_flags(flags)

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
        scale_CH_est_i = scale_CH_est * (1 - INT['ANC']['b']) * INT['mcomp']['Anemia_PPH']['b'] * \
                         INT['mcomp']['Anemia_APH']['b'] * INT['mcomp']['PP_PPH']['b'] * INT['mcomp']['PPH']['b']
        scale_CL_est_i = scale_CL_est * (1 - INT['ANC']['b']) * INT['mcomp']['Anemia_PPH']['b'] * \
                         INT['mcomp']['Anemia_APH']['b'] * INT['mcomp']['PPH']['b']

        for j in range(SC['n']):  # for each sub county
            transfer = (1 - SC['Class'][j]) + INT['trans']['b'] * SC['Class'][j]  # p2
            if i > 0:  # skip time 1
                SC['LB1s'][j][i, :] = f_INT_LB_effect(SC['LB1s'][j][i - 1, :], INT, False, i, Capacity_ratio1[:, j])
                # compute pushback (overcapacity => reduced CHV effect)
                SC['LB2s'][j][i, :], Push_back[i, j] = f_INT_LB_effect(SC['LB2s'][j][i - 1, :], INT, True, i,
                                                                       Capacity_ratio2[:, j])

                if flags[0]:
                    LB_tot_j = np.maximum(SC['LB1s'][j][0, :], 1)  # (zero LB in L5 for sub 2)
                    SDR_multiplier_1_j = SC['LB1s'][j][i, :] / LB_tot_j  # ratio of sums = 1
                    SDR_multiplier_2_j = SC['LB2s'][j][i, :] / LB_tot_j

                    n_MM1s_H, n_nM1s_H, _ = f_MM(LB_tot_j, SDR_multiplier_1_j, scale_CH_est_i, n_CHs['refer'][j],
                                                 q_CH_transfer_from * transfer, p_MM[1], p_MM_scale_transfer, INT)
                    n_MM2s_H, n_nM2s_H, _ = f_MM(LB_tot_j, SDR_multiplier_2_j, scale_CH_est_i, n_CHs['refer'][j],
                                                 q_CH_transfer_from * transfer, p_MM[1], p_MM_scale_transfer, INT)
                    n_MM1s_L, n_nM1s_L, _ = f_MM(LB_tot_j, SDR_multiplier_1_j, scale_CL_est_i, n_CLs['refer'][j],
                                                 q_CL_transfer_from * transfer, p_MM[0], p_MM_scale_transfer, INT)
                    n_MM2s_L, n_nM2s_L, _ = f_MM(LB_tot_j, SDR_multiplier_2_j, scale_CL_est_i, n_CLs['refer'][j],
                                                 q_CL_transfer_from * transfer, p_MM[0], p_MM_scale_transfer, INT)

                    n_MMs[i, :, j, 0] = n_MM1s_H + n_MM1s_L
                    n_MMs[i, :, j, 1] = n_MM2s_H + n_MM2s_L
                    p_MMs[i, :, j, :] = n_MMs[i, :, j, :] / LB_tot_j[:, None]

                    n_nMs[i, :, j, 0] = n_nM1s_H + n_nM1s_L
                    n_nMs[i, :, j, 0] = n_nM2s_H + n_nM2s_L
                    p_nMs[i, :, j, :] = n_nMs[i, :, j, :] / LB_tot_j[:, None]

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

    pph_H = scale_CH_est / (INT['mcomp']['PPH']['inc'] * INT['mcomp']['Anemia_PPH']['b'] * INT['mcomp']['PP_PPH']['b'] *
                            (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[n_months - 1])
    pph_L = scale_CL_est / (INT['mcomp']['PPH']['inc'] * INT['mcomp']['Anemia_PPH']['b'] *
                            (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[n_months - 1])

    outcomes_dict['Maternal Deaths'] = np.sum(n_MM[i, :, 0])
    outcomes_dict['Neonatal Deaths'] = np.sum(n_nM[i, :, 0])
    outcomes_dict['High PPH'] = (np.sum(n_stats_H1['Initial Complications Post Referral'][n_months - 1]) - np.sum(
        n_MM1_H)) / pph_H
    outcomes_dict['Low PPH'] = (np.sum(n_stats_L1['Initial Complications Post Referral'][n_months - 1]) - np.sum(
        n_MM1_L)) / pph_L

    # still working on
    return SC, n_MM, n_nM, n_MMs, n_nMs, n_stats_H1, n_stats_H2, n_stats_L1, n_stats_L2, outcomes_dict

flags = [flag_sub, flag_CHV, flag_ANC, flag_ANC2, flag_refer, flag_trans, flag_int1, flag_int2, flag_int3, flag_int4, flag_trans2]
b_flags = [1,0,0,0,0,0,0,0,0,0,0]
bSC, bn_MM, bn_nM, bn_MMs, bn_nMs, bn_stats_H1, bn_stats_H2, bn_stats_L1, bn_stats_L2, boutcomes = run_model(b_flags) # always baseline
iSC, n_MM, n_nM, n_MMs, n_nMs, n_stats_H1, n_stats_H2, n_stats_L1, n_stats_L2, outcomes = run_model(flags)                       # intervention, can rename                     # intervention, can rename

###############PLOTING HERE####################
# Normalize data values to the colormap range
cmap = plt.get_cmap('viridis')
data=[0,1,2,3]
normalize = plt.Normalize(min(data), max(data))
colors = [cmap(normalize(value)) for value in data]
labels = ['Home', 'L23', 'L4', 'L5']

if selected_plotA == "Live births":
    #bSC, bn_MM, bn_nM, bn_MMs, bn_nMs, bn_stats_H1, bn_stats_H2, bn_stats_L1, bn_stats_L2, boutcomes = run_model(b_flags)
    overall_lbs = np.zeros((n_months,4))
    for j in range(4):
        tots=np.zeros((n_months))
        for i in range(12):
            tots += bSC['LB2s'][i][:,j]
        overall_lbs[:,j] = tots

    fig,axes = plt.subplots(1,2,figsize=(12,4))
    #iSC, n_MM, n_nM, n_MMs, n_nMs, n_stats_H1, n_stats_H2, n_stats_L1, n_stats_L2, outcomes = run_model(flags)
    overall_lbs_i = np.zeros((n_months,4))
    for j in range(4):
        tots=np.zeros((n_months))
        for i in range(12):
            tots += iSC['LB2s'][i][:,j]
        overall_lbs_i[:,j] = tots

    for j in range(4):
        axes[0].plot(range(n_months), overall_lbs[:,j], color=colors[j])
        axes[1].plot(range(n_months), overall_lbs_i[:,j], color=colors[j])
        axes[0].set_title('Baseline')
        axes[1].set_title('Intervention')
        # axes[k].set_title(f'{p_title[k]}')
    for ax in axes:
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Live Births')
        ax.set_ylim([0,35000])

    plt.suptitle('Live Births')
    fig.legend([f'{labels[j]}' for j in range(4)])
    st.pyplot(plt)


if selected_plotA == "Neonatal deaths":
    # categories = ['Asphyxia', 'Sepsis', 'Preterm', 'Other']
    #
    # fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12, 8))
    # axs = axs.flatten()
    # for i in range(12):
    #     asphyxia = np.sum(np.sum(bn_nMs[35, :, :, :], axis=2), axis=0)[i] * INT['mort']['n_asphyxia']
    #     sepsis = np.sum(np.sum(bn_nMs[35, :, :, :], axis=2), axis=0)[i] * INT['mort']['n_sepsis']
    #     preterm = np.sum(np.sum(n_nMs[35, :, :, :], axis=2), axis=0)[i] * INT['mort']['n_preterm'] * \
    #               INT['ncomp']['preterm']['b']
    #     other = np.sum(np.sum(n_nMs[35, :, :, :], axis=2), axis=0)[i] - asphyxia - sepsis - preterm
    #     axs[i].bar(categories, [asphyxia, sepsis, preterm, other], color='purple')
    #     axs[i].set_title(f'{subcounties[i]}')
    #
    # plt.ylabel('Deaths')
    # plt.suptitle('Neonatal deaths by Type')
    # #plt.tight_layout()
    # st.pyplot(plt)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    save_1 = np.sum(bn_nM, axis=2)
    save_2 = np.sum(n_nM, axis=2)
    p_title = ['Baseline', 'Intervention']

    for i in range(2):
        for j in range(4):
            if i == 0:
                axes[i].plot(range(n_months), save_1[:, j], color=colors[j])
            else:
                axes[i].plot(range(n_months), save_2[:, j], color=colors[j])
        # axes[i].set_ylim([0,1000])
        axes[i].set_xlabel('Time (months)')
        axes[i].set_ylabel('Deaths')
        axes[i].set_title(f'{p_title[i]}')
        axes[i].set_ylim([0, 1500])

    plt.suptitle('Neonatal Mortalities')
    fig.legend([f'{labels[j]}' for j in range(4)])
    st.pyplot(plt)

if selected_plotA == "Maternal deaths":
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    p_title = ['Baseline', 'Intervention']
    save_3 = np.sum(bn_MM, axis=2)
    save_4 = np.sum(n_MM, axis=2)

    for i in range(2):
        for j in range(4):
            if i == 0:
                axes[i].plot(range(n_months), save_3[:, j], color=colors[j])
            else:
                axes[i].plot(range(n_months), save_4[:, j], color=colors[j])
        # axes[i].set_ylim([0,1000])
        # axes[i].set_ylim([0,1000])
        axes[i].set_xlabel('Time (months)')
        axes[i].set_ylabel('Deaths')
        axes[i].set_title(f'{p_title[i]}')
        axes[i].set_ylim([0, 900])

    plt.suptitle('Maternal Mortalities')
    fig.legend([f'{labels[j]}' for j in range(4)])
    st.pyplot(plt)

if selected_plotA == "MMR":
    tots=[]

    for j in range(n_months):
        sums=0
        for key,val in SC['LB2s'].items():
            sums += np.sum(val[j, 2:4])
        tots.append(sums)
    p_l45 = tots/np.sum(LB_tot)
    MMR = np.sum(np.sum(n_MM, axis=1), axis=1)/np.sum(LB_tot)
    # Plotting the bar graph
    plt.bar(range(n_months), p_l45, color='darkblue', label='Bar Graph')

    # Plotting the line graph on top
    plt.plot(range(n_months), MMR*50, marker='o', color='orange', label='Line Graph')
    plt.xlabel('Months')
    plt.ylabel('Percent')
    plt.title('Trends in Percent Live Births at L4/5 and MMR')
    plt.legend(['MMR rate, scaled by 50', 'Percent live births at L4/5'])

    st.pyplot(plt)

if selected_plotA == "DALYs":
    categories = list(outcomes.keys())
    values1 = list(boutcomes.values())
    values2 = list(outcomes.values())

    bar_width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax in axes:
        ax.set_ylim([0, 8000])

    axes[0].bar(categories, values1, width=bar_width, label='Baseline', color='blue')
    axes[0].set_xlabel('Morbidities and Mortalities')
    axes[1].set_ylabel('DALYs')
    axes[0].set_title('Baseline')
    axes[1].bar(categories, values2, width=bar_width, label='Intervention', color='orange')
    axes[1].set_xlabel('Morbidities and Mortalities')
    axes[1].set_ylabel('DALYs')
    axes[1].set_title('Intervention')
    plt.suptitle('DALYs by Severe Health Outcomes')
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


def get_ce(boutcomes, bn_stats_H1, bn_stats_H2, bn_stats_L1, bn_stats_L2, outcomes, flags):
    """get cost per DALY averted"""
    cost = 0
    total = 0
    for x, y in zip(boutcomes.values(), outcomes.values()):
        total += (x - y)

    INT = set_flags(flags)

    pph_effect_end_H = (scale_CH_est / (INT['mcomp']['PPH']['inc'] * INT['mcomp']['Anemia_PPH']['b'] *  INT['mcomp']['PP_PPH']['b'] * (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[n_months-1]* INT['mcomp']['PPH']['b']))
    pph_effect_end_L = (scale_CL_est / (INT['mcomp']['PPH']['inc'] * INT['mcomp']['Anemia_PPH']['b'] *
                                        (1 - INT['ANC']['bs'] * (INT['ANC']['ANC_PPH']['b']))[n_months - 1]))

    aph_effect_end = (scale_CH_est / (INT['mcomp']['APH']['inc'] * INT['mcomp']['Anemia_APH']['b']))
    no_effect = (scale_CH_est / INT['mcomp']['APH']['inc'])

    bno_push_L = np.concatenate(bn_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
    bno_push_H = np.concatenate(bn_stats_H1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))

    bpush_L = np.concatenate(bn_stats_L2['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
    bpush_H = np.concatenate(bn_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
    bpush = bpush_L + bpush_H

    if flags[6]:
        comps = np.sum(
            bpush_L[n_months - 1, :] / pph_effect_end_L + bpush_H[n_months - 1, :] / pph_effect_end_H + bpush[
                                                                                                        n_months - 1,
                                                                                                        :] / aph_effect_end)
        cost += comps
    if flags[8]:
        cost += 2000 * 232  # number of L2/3 facilities with provided ANC
    if flags[9]:
        cost += 53.68 * np.sum(
            LB_tot)  # consider, does this actually happen across all facility levels (??) or only the mothers who receive

    return cost / total

if selected_plotA == "Cost effectiveness":
    _, _, _, _, _, bn_stats_H1, bn_stats_H2, bn_stats_L1, bn_stats_L2, boutcomes = run_model(
        b_flags)
    _, _, _, _, _, _, _, _, _, outcomes1 = run_model([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])  # Obstetric drape
    _, _, _, _, _, _, _, _, _, outcomes3 = run_model([1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])  # Ultrasound
    _, _, _, _, _, _, _, _, _, outcomes4 = run_model([1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0])  # antenatal corticosteroids
    _, _, _, _, _, _, _, _, _, outcomesc = run_model([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])  # combined
    cost_effect = np.array([get_ce(boutcomes, bn_stats_H1, bn_stats_H2, bn_stats_L1, bn_stats_L2, outcomes1,
                                   [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                            get_ce(boutcomes, bn_stats_H1, bn_stats_H2, bn_stats_L1, bn_stats_L2, outcomes3,
                                   [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]),
                            get_ce(boutcomes, bn_stats_H1, bn_stats_H2, bn_stats_L1, bn_stats_L2, outcomes4,
                                   [1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]),
                            get_ce(boutcomes, bn_stats_H1, bn_stats_H2, bn_stats_L1, bn_stats_L2, outcomesc,
                                   [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])])

    plt.bar(['Obstetric Drape', 'Ultrasound', 'Antenatal Drugs', 'Combined'], cost_effect, color='orange')
    plt.xlabel('Interventions')
    plt.ylabel('USD per DALYs averted')
    plt.title('Cost-effectiveness of interventions')
    for i, value in enumerate(cost_effect):
        plt.text(i, value + 0.5, str(round(value)), ha='center', va='bottom')

    st.pyplot(plt)

### referrals related
brno_push = np.concatenate(bn_stats_L1['Referrals'].to_numpy()).reshape((n_months, 4)) + np.concatenate(bn_stats_H1['Referrals'].to_numpy()).reshape((n_months, 4))
brpush = np.concatenate(bn_stats_L2['Referrals'].to_numpy()).reshape((n_months, 4)) + np.concatenate(bn_stats_L1['Referrals'].to_numpy()).reshape((n_months, 4))

rno_push = np.concatenate(n_stats_L1['Referrals'].to_numpy()).reshape((n_months, 4)) + np.concatenate(n_stats_H1['Referrals'].to_numpy()).reshape((n_months, 4))
rpush = np.concatenate(n_stats_L2['Referrals'].to_numpy()).reshape((n_months, 4)) + np.concatenate(n_stats_L1['Referrals'].to_numpy()).reshape((n_months, 4))

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

# ### APH Hemorrhage ###
# bpush = bpush_L + bpush_H
# push = push_L + push_H
#
# aph_effect_beg = (scale_CH_est / (INT['comp']['APH']['inc'] * INT['comp']['Anemia_APH']['b']))
# aph_effect_end = (scale_CH_est / (INT['comp']['APH']['inc'] * INT['comp']['Anemia_APH']['b']))
# no_effect = (scale_CH_est / INT['comp']['APH']['inc'])
#
# to_plotbeg = np.vstack((push[0,:] / no_effect, push[0,:] / aph_effect_beg))
# to_plotend = np.vstack((push[n_months-1,:] / no_effect, push[n_months-1,:] / aph_effect_end))
#
# if selected_plotA == "APH complications":
#     fig, axes = plt.subplots(1,2,figsize = (12,4))
#     for i in range(2):
#         categories = ['Home', 'L2/3', 'L4', 'L5']
#         if i==1:
#             values1 = np.clip(list(to_plotend[0,:]), 0, None)
#             values2 = np.clip(list(to_plotend[1,:]), 0, None)
#             # values1 = to_plotend[0,:]
#             # values2 = to_plotend[1,:]
#         else:
#             values1 = np.clip(list(to_plotbeg[0,:]), 0, None)
#             values2 = np.clip(list(to_plotbeg[1,:]), 0, None)
#             # values1 = to_plotbeg[0,:]
#             # values2 = to_plotbeg[1,:]
#
#         # Set the width of the bars
#         bar_width = 0.35
#
#         # Set the positions of the bars on the x-axis
#         bar_positions1 = np.arange(len(categories))
#         bar_positions2 = bar_positions1 + bar_width
#
#     # Create the bar graph
#         axes[i].bar(bar_positions1, values1, width=bar_width, label='Group 1')
#         axes[i].bar(bar_positions2, values2, width=bar_width, label='Group 2')
#         axes[i].set_xticks(bar_positions1 + bar_width / 2, categories)
#         if i ==1:
#             axes[i].set_title('APH Complications by Facility Level (32 months)')
#         else:
#             axes[i].set_title('APH Complications by Facility Level (0 months)')
#
#     #plt.tight_layout()
#     plt.legend(['Baseline', 'With Intervention'])
#     st.pyplot(fig)

### PPH Hemorrhage ###
if selected_plotA == "PPH complications":
    ### PPH Hemorrhage ###
    iINT = set_flags(flags)
    bpush_L = np.concatenate(bn_stats_L2['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
    bpush_H = np.concatenate(bn_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape((n_months, 4))
    bpush = bpush_L + bpush_H
    push = push_L + push_H

    pph_effect_beg_H = (scale_CH_est / (
                iINT['mcomp']['PPH']['inc'] * iINT['mcomp']['Anemia_PPH']['b'] * iINT['mcomp']['PP_PPH']['b'] *
                (1 - iINT['ANC']['bs'] * (iINT['ANC']['ANC_PPH']['b']))[0] * iINT['mcomp']['PPH']['b']))
    pph_effect_end_H = (scale_CH_est / (
                iINT['mcomp']['PPH']['inc'] * iINT['mcomp']['Anemia_PPH']['b'] * iINT['mcomp']['PP_PPH']['b'] *
                (1 - iINT['ANC']['bs'] * (iINT['ANC']['ANC_PPH']['b']))[n_months - 1] * iINT['mcomp']['PPH']['b']))
    pph_effect_beg_L = (scale_CL_est / (iINT['mcomp']['PPH']['inc'] * iINT['mcomp']['Anemia_PPH']['b'] *
                                        (1 - iINT['ANC']['bs'] * (iINT['ANC']['ANC_PPH']['b']))[0]))
    pph_effect_end_L = (scale_CL_est / (iINT['mcomp']['PPH']['inc'] * iINT['mcomp']['Anemia_PPH']['b'] *
                                        (1 - iINT['ANC']['bs'] * (iINT['ANC']['ANC_PPH']['b']))[n_months - 1]))

    no_effect = (scale_CH_est / iINT['mcomp']['PPH']['inc'])

    to_plotbeg = np.vstack(
        (bpush[0, :] / no_effect, bpush_L[0, :] / pph_effect_beg_L + bpush_H[0, :] / pph_effect_beg_H))
    to_plotend = np.vstack((bpush[n_months - 1, :] / no_effect,
                            bpush_L[n_months - 1, :] / pph_effect_end_L + bpush_H[n_months - 1, :] / pph_effect_end_H))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(2):
        categories = ['Home', 'L2/3', 'L4', 'L5']
        if i == 1:
            # values1 = np.clip(list(to_plotend[0,:]), 0, None)
            # values2 = np.clip(list(to_plotend[1,:]), 0, None)
            values1 = to_plotend[0, :]
            values2 = to_plotend[1, :]
        else:
            # values1 = np.clip(list(to_plotbeg[0,:]), 0, None)
            # values2 = np.clip(list(to_plotbeg[1,:]), 0, None)
            values1 = to_plotbeg[0, :]
            values2 = to_plotbeg[1, :]

        # Set the width of the bars
        bar_width = 0.35

        # Set the positions of the bars on the x-axis
        bar_positions1 = np.arange(len(categories))
        bar_positions2 = bar_positions1 + bar_width

        # Create the bar graph
        axes[i].bar(bar_positions1, values1, width=bar_width, label='Group 1')
        axes[i].bar(bar_positions2, values2, width=bar_width, label='Group 2')
        axes[i].set_xticks(bar_positions1 + bar_width / 2, categories)
        if i == 1:
            axes[i].set_title(f'PPH Complications by Facility Level - {n_months} months')
        else:
            axes[i].set_title('PPH Complications by Facility Level - 1 month')

    #plt.tight_layout()
    plt.legend(['Baseline', 'With Intervention'])
    st.pyplot(fig)

if selected_plotA == "Effects of CHV pushback":
    no_push = np.concatenate(n_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape(
        (n_months, 4)) + np.concatenate(n_stats_H1['Initial Complications Post Referral'].to_numpy()).reshape(
        (n_months, 4))
    push = np.concatenate(n_stats_L2['Initial Complications Post Referral'].to_numpy()).reshape(
        (n_months, 4)) + np.concatenate(n_stats_L1['Initial Complications Post Referral'].to_numpy()).reshape(
        (n_months, 4))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
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

if selected_plotA == "Effects of CHV pushback":
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

if selected_plotB == "Maternal deaths":
    st.markdown("<h3 style='text-align: left;'>Maternal deaths by subcounty and facility level</h3>",
                unsafe_allow_html=True)
    options = ['Home', 'L2/3', 'L4', 'L5']
    selected_options = st.multiselect('Select levels', options)

    num_rows, num_cols = 4, 3
    plot_columns = [st.columns(num_cols) for _ in range(num_rows)]
    for j in range(12):
        df = pd.DataFrame(np.sum(n_MMs[1:, :, :, :], axis = 3)[:,:,j])
        df['month'] = range(1, 36)
        df = df.melt(id_vars=['month'], var_name='level', value_name='value')
        # Mapping dictionary
        mapping = {
            0: 'Home',
            1: 'L2/3',
            2: 'L4',
            3: 'L5'
        }
        # Replace values in the 'Levels' column
        df['level'] = df['level'].replace(mapping)
        df = df[df['level'].isin(selected_options)]
        chart = (
            alt.Chart(
                data=df,
                title=f'{subcounties[j]}',
            )
            .mark_line()
            .encode(
                x=alt.X("month", axis=alt.Axis(title="Month")),
                y=alt.Y("value", axis=alt.Axis(title="# of Deaths")),
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

if selected_plotB == "Neonatal deaths":
    # arr = np.sum(n_nMs[1:, :, :, :], axis=3)
    # fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12, 8))
    # axs = axs.flatten()
    # # Iterate over each row in the array
    # for i in range(arr.shape[2]):
    #     # Iterate over each subplot in the row
    #     for j in range(arr.shape[1]):
    #         # Plot the line for each subplot
    #         if j > 0:
    #             axs[i].plot(range(35), arr[:, j, i], label=f'Line {j + 1}')
    #
    #     # Set subplot title and labels
    #     axs[i].set_title(f'{subcounties[i]}')
    #     axs[i].set_xlabel('Months')
    #     axs[i].set_ylabel('Deaths')
    #
    # plt.suptitle('Neonatal Mortalities by Subcounty, excluding home')
    # st.pyplot(fig)
    # # Adjust layout to prevent clipping of titles
    # #plt.tight_layout()
    # # Show the plot
    # #plt.show()
    st.markdown("<h3 style='text-align: left;'>Neonatal deaths by subcounty and facility level</h3>",
                unsafe_allow_html=True)
    options = ['Home', 'L2/3', 'L4', 'L5']
    selected_options = st.multiselect('Select levels', options)

    num_rows, num_cols = 4, 3
    plot_columns = [st.columns(num_cols) for _ in range(num_rows)]
    for j in range(12):
        df = pd.DataFrame(np.sum(n_nMs[1:, :, :, :], axis = 3)[:,:,j])
        df['month'] = range(1, 36)
        df = df.melt(id_vars=['month'], var_name='level', value_name='value')
        # Mapping dictionary
        mapping = {
            0: 'Home',
            1: 'L2/3',
            2: 'L4',
            3: 'L5'
        }
        # Replace values in the 'Levels' column
        df['level'] = df['level'].replace(mapping)
        df = df[df['level'].isin(selected_options)]
        chart = (
            alt.Chart(
                data=df,
                title=f'{subcounties[j]}',
            )
            .mark_line()
            .encode(
                x=alt.X("month", axis=alt.Axis(title="Month")),
                y=alt.Y("value", axis=alt.Axis(title="# of Deaths")),
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

if selected_plotB == "Live births":
    #Streamlit Version
    st.markdown("<h3 style='text-align: center;'>Live births by subcounty and facility level</h3>",
                unsafe_allow_html=True)
    options = ['Home', 'L2/3', 'L4', 'L5']
    selected_options = st.multiselect('Select levels', options)

    num_rows, num_cols = 4, 3
    plot_columns = [st.columns(num_cols) for _ in range(num_rows)]

    for j in range(12):
        df = pd.DataFrame(iSC['LB1s'][j])
        df['month'] = range(1, 37)
        df = df.melt(id_vars=['month'], var_name='level', value_name='value')
        # Mapping dictionary
        mapping = {
            0: 'Home',
            1: 'L2/3',
            2: 'L4',
            3: 'L5'
        }
        # Replace values in the 'Levels' column
        df['level'] = df['level'].replace(mapping)
        # Filter the DataFrame based on selected options
        df = df[df['level'].isin(selected_options)]

        chart = (
            alt.Chart(
                data=df,
                title=f'{subcounties[j]}',
            )
            .mark_line()
            .encode(
                x=alt.X("month", axis=alt.Axis(title="Month")),
                y=alt.Y("value", axis=alt.Axis(title="# of Births")),
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
    # num_rows, num_cols = 4, 3
    #
    # # Create a grid of subplots
    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    #
    # # Iterate over j from 0 to 11
    # for j in range(num_rows * num_cols):
    #     # Replace 7 with j to access different LB1s
    #     y_values = SC['LB1s'][j]
    #     y_values = y_values.T
    #
    #     # Calculate the row and column indices for the current subplot
    #     row, col = divmod(j, num_cols)
    #
    #     # Plot the data in the current subplot
    #     for i in range(4):
    #         axes[row, col].plot(range(n_months), y_values[i])
    #
    #     axes[row, col].set_title(f'Subcounty {j + 1}')
    #     axes[row, col].set_xlabel('Months')
    #     axes[row, col].set_ylabel('# of Births')
    #
    # fig.suptitle('Live births by subcounty and facility level')
    # #plt.tight_layout()
    # #plt.show()
    # st.pyplot(fig)

if select_level == "Subcounty in Map":
    shapefile_path2 = 'ke_subcounty.shp'
    data2 = gpd.read_file(shapefile_path2)
    data2 = data2.loc[data2['county'] == 'Kakamega',]
    data2['subcounty'] = data2['subcounty'].str.split().str[0]
    LB_matrix = np.zeros((12, 4))
    MM_matrix = np.zeros((12, 4))
    NM_matrix = np.zeros((12, 4))
    MMR_matrix = np.zeros((12, 4))
    NMR_matrix = np.zeros((12, 4))
    bLB_matrix = np.zeros((12, 4))
    bMM_matrix = np.zeros((12, 4))
    bNM_matrix = np.zeros((12, 4))
    bMMR_matrix = np.zeros((12, 4))
    bNMR_matrix = np.zeros((12, 4))
    for i in range(12):
        LB_matrix[i, :] = iSC['LB1s'][i][35, :]
        MM_matrix[i, :] = np.sum(n_MMs, axis=3)[35,:,i]
        NM_matrix[i, :] = np.sum(n_nMs, axis=3)[35,:,i]
        bLB_matrix[i, :] = bSC['LB1s'][i][35, :]
        bMM_matrix[i, :] = np.sum(bn_MMs, axis=3)[35,:,i]
        bNM_matrix[i, :] = np.sum(bn_nMs, axis=3)[35,:,i]
        for j in range(4):
            MMR_matrix[i, j] = MM_matrix[i, j] / LB_matrix[i, j] * 1000
            NMR_matrix[i, j] = NM_matrix[i, j] / LB_matrix[i, j] * 1000
            bMMR_matrix[i, j] = bMM_matrix[i, j] / bLB_matrix[i, j] * 1000
            bNMR_matrix[i, j] = bNM_matrix[i, j] / bLB_matrix[i, j] * 1000

    MMR_matrix[np.isinf(MMR_matrix)] = 0
    NMR_matrix[np.isinf(NMR_matrix)] = 0
    bMMR_matrix[np.isinf(MMR_matrix)] = 0
    bNMR_matrix[np.isinf(NMR_matrix)] = 0
    df_MMR = pd.DataFrame(MMR_matrix)
    df_NMR = pd.DataFrame(NMR_matrix)
    df_LB = pd.DataFrame(LB_matrix)
    df_bMMR = pd.DataFrame(bMMR_matrix)
    df_bNMR = pd.DataFrame(bNMR_matrix)
    df_bLB = pd.DataFrame(bLB_matrix)
    new_column_names_MMR = ['MMR_Home', 'MMR_L23', "MMR_L4", "MMR_L5"]
    new_column_names_NMR = ['NMR_Home', 'NMR_L23', "NMR_L4", "NMR_L5"]
    new_column_names_LB = ['LB_Home', 'LB_L23', "LB_L4", "LB_L5"]
    new_column_names_bMMR = ['bMMR_Home', 'bMMR_L23', "bMMR_L4", "bMMR_L5"]
    new_column_names_bNMR = ['bNMR_Home', 'bNMR_L23', "bNMR_L4", "bNMR_L5"]
    new_column_names_bLB = ['bLB_Home', 'bLB_L23', "bLB_L4", "bLB_L5"]
    df_MMR.columns = new_column_names_MMR
    df_NMR.columns = new_column_names_NMR
    df_LB.columns = new_column_names_LB
    df_bMMR.columns = new_column_names_bMMR
    df_bNMR.columns = new_column_names_bNMR
    df_bLB.columns = new_column_names_bLB
    df_MMR['Sum_MMR'] = df_MMR.iloc[:, 0:4].sum(axis=1)
    df_NMR['Sum_NMR'] = df_NMR.iloc[:, 0:4].sum(axis=1)
    df_LB['Sum_LB'] = df_LB.iloc[:, 0:4].sum(axis=1)
    df_bMMR['Sum_bMMR'] = df_bMMR.iloc[:, 0:4].sum(axis=1)
    df_bNMR['Sum_bNMR'] = df_bNMR.iloc[:, 0:4].sum(axis=1)
    df_bLB['Sum_bLB'] = df_bLB.iloc[:, 0:4].sum(axis=1)
    df_MMR['subcounty'] = subcounties
    df_NMR['subcounty'] = subcounties
    df_LB['subcounty'] = subcounties
    df_bMMR['subcounty'] = subcounties
    df_bNMR['subcounty'] = subcounties
    df_bLB['subcounty'] = subcounties
    data2 = data2.merge(df_MMR, left_on='subcounty', right_on='subcounty')
    data2 = data2.merge(df_NMR, left_on='subcounty', right_on='subcounty')
    data2 = data2.merge(df_LB, left_on='subcounty', right_on='subcounty')
    data2 = data2.merge(df_bMMR, left_on='subcounty', right_on='subcounty')
    data2 = data2.merge(df_bNMR, left_on='subcounty', right_on='subcounty')
    data2 = data2.merge(df_bLB, left_on='subcounty', right_on='subcounty')
    if selected_plotC == "MMR":
        st.markdown("<h3 style='text-align: left;'>Maternal deaths per 1,000 live births</h3>",
                    unsafe_allow_html=True)

        fig1 = px.choropleth_mapbox(data2,
                               geojson=data2.geometry,
                               locations=data2.index,
                               mapbox_style="open-street-map",
                               color=data2.Sum_bMMR,
                               center={'lon': 34.785511, 'lat': 0.430930},
                               zoom=8,
                               color_continuous_scale="ylgnbu",  # You can choose any color scale you prefer
                               range_color=(20, 70)  # Define the range of colors
                                   )

        fig2 = px.choropleth_mapbox(data2,
                               geojson=data2.geometry,
                               locations=data2.index,
                               mapbox_style="open-street-map",
                               color=data2.Sum_MMR,
                               center={'lon': 34.785511, 'lat': 0.430930},
                               zoom=8,
                               color_continuous_scale="ylgnbu",  # You can choose any color scale you prefer
                               range_color=(20, 70)  # Define the range of colors
                                   )
        col1, col2 = st.columns(2)
        with col1:
            st.write('Baseline')
            st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
        with col2:
            st.write('Intervention')
            st.plotly_chart(fig2, theme="streamlit", use_container_width=True)


    if selected_plotC == "NMR":
        st.markdown("<h3 style='text-align: left;'>Neonatal deaths per 1,000 live births</h3>",
                    unsafe_allow_html=True)

        fig1 = px.choropleth_mapbox(data2,
                                   geojson=data2.geometry,
                                   locations=data2.index,
                                   mapbox_style="open-street-map",
                                   color=data2.Sum_bNMR,
                                   center={'lon': 34.785511, 'lat': 0.430930},
                                   zoom=8,
                                   color_continuous_scale="ylgnbu",  # You can choose any color scale you prefer
                                   range_color=(200, 350)  # Define the range of colors
                                   )

        fig2 = px.choropleth_mapbox(data2,
                                   geojson=data2.geometry,
                                   locations=data2.index,
                                   mapbox_style="open-street-map",
                                   color=data2.Sum_NMR,
                                   center={'lon': 34.785511, 'lat': 0.430930},
                                   zoom=8,
                                   color_continuous_scale="ylgnbu",  # You can choose any color scale you prefer
                                   range_color=(200, 350)  # Define the range of colors
                                   )
        col1, col2 = st.columns(2)
        with col1:
            st.write('Baseline')
            st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
        with col2:
            st.write('Intervention')
            st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

    if selected_plotC == "% deliveries in L4/5":
        st.markdown("<h3 style='text-align: left;'>% deliveries in L4/5</h3>",
                    unsafe_allow_html=True)
        boutcome = (data2.bLB_L4 + data2.bLB_L5) / data2.Sum_bLB
        outcome = (data2.LB_L4 + data2.LB_L5) / data2.Sum_LB
        fig1 = px.choropleth_mapbox(data2,
                                   geojson=data2.geometry,
                                   locations=data2.index,
                                   mapbox_style="open-street-map",
                                   color= boutcome,
                                   center={'lon': 34.785511, 'lat': 0.430930},
                                   zoom=8,
                                   color_continuous_scale="ylgnbu",  # You can choose any color scale you prefer
                                   range_color=(0, 1)  # Define the range of colors
                                   )
        fig2 = px.choropleth_mapbox(data2,
                                    geojson=data2.geometry,
                                    locations=data2.index,
                                    mapbox_style="open-street-map",
                                    color=outcome,
                                    center={'lon': 34.785511, 'lat': 0.430930},
                                    zoom=8,
                                    color_continuous_scale="ylgnbu",  # You can choose any color scale you prefer
                                    range_color=(0, 1)  # Define the range of colors
                                    )
        col1, col2 = st.columns(2)
        with col1:
            st.write('Baseline')
            st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
        with col2:
            st.write('Intervention')
            st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

##############Old plots in Streamlit version##################
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
# if selected_plotA == "Live births":
    # to_plotbeg = np.vstack((brpush[0,:], rpush[0,:]))
    # to_plotend = np.vstack((brpush[n_months-1,:], rpush[n_months-1,:]))
    # num_rows, num_cols = 1, 2
    # plot_columns = [st.columns(num_cols) for _ in range(num_rows)]
    #
    # ymin = min(np.min(to_plotend), np.min(to_plotbeg)) - 100
    # ymax = max(np.max(to_plotend), np.max(to_plotbeg)) + 100
    # for j in range(2):
    #     if j == 0:
    #         df = to_plotbeg
    #         month = 1
    #     else:
    #         df = to_plotend
    #         month = 36
    #     data = pd.DataFrame({
    #         'Bar': ['Home', 'L2/3', 'L4', 'L5'],
    #         'Baseline': df.T[:,0],
    #         'Intervention': df.T[:,1]
    #     }).melt(id_vars=['Bar'], var_name='Group', value_name='Value')
    #     chart = alt.Chart(
    #         data,
    #         title = f'Change of live births at month {month}',
    #     ).mark_bar().encode(
    #         x=alt.X('Group:O', axis=alt.Axis(title=None, labels=False, ticks=False)),
    #         y=alt.Y('Value:Q', title='Live births', axis=alt.Axis(grid=True)).scale(domain=(ymin, ymax)),
    #         color=alt.Color('Group:N', scale=alt.Scale(domain=['Baseline', 'Intervention'], range=['blue', 'orange'])),
    #         column=alt.Column('Bar:N', header=alt.Header(title=None, labelOrient='bottom')),
    #     ).configure_view(
    #     stroke='transparent'
    #     ).configure_title(
    #       anchor='middle'
    #    )
    #
    #     row = j // num_cols
    #     col = j % num_cols
    #
    #     with plot_columns[row][col]:
    #         st.altair_chart(chart)
