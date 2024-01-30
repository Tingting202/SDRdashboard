import streamlit as st
from matplotlib import pyplot as plt
import geopandas as gpd
import numpy as np
from sympy import symbols, Eq, solve
import plotly.express as px
import pandas as pd
import altair as alt
import math
import plotly.graph_objs as go
#from SDR_Dec_streamlit_functions import run_model, get_aggregate, get_cost_effectiveness
from scipy.optimize import fsolve
import time
import cProfile

st.set_page_config(layout="wide")
options = {
    'Off': 0,
    'On': 1
}

selected_plotA = None
selected_plotB = None
selected_plotC = None
flag_CHV = 0
flag_ANC = 0
flag_ANC_time = 0
flag_int1 = 0
flag_int2 = 0
flag_int3 = 0
flag_int4 = 0
flag_int5 = 0
flag_int6 = 0
flag_trans = 0
flag_refer = 0
flag_sdr = 0
flag_capacity = 0
flag_community = 0
flag_refer_capacity = 0
diagnosisrate = 0
ultracovhome = 0
CHV_cov = 0
CHV_45 = 0
CHV_ANC = 0
CHV_pushback = False
referadded = 0
transadded = 0
capacity_added = 0
know_added = 0
supply_added = 0
supply_added_INT1 = 0
supply_added_INT2 = 0
supply_added_INT5 = 0
supply_added_INT6 = 0
ANCadded = 0
refer_capacity = 100


Cap_b = 1
QOC_b = 1

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

SC_ID = {
    0: "Butere",
    1: "Ikolomani",
    2: "Khwisero",
    3: "Likuyani",
    4: "Lugari",
    5: "Lurambi",
    6: "Malava",
    7: "Matungu",
    8: "Mumias East",
    9: "Mumias West",
    10: "Navakholo",
    11: "Shinyalu"
}

def reset_global_vars():
    global_vars = {
        'ANCadded': 0,
        'CHV_pushback': False,
        'CHV_cov': 0,
        'CHV_45': 0,
        'know_added': 0,
        'supply_added': {'supply_added_INT1': 0,
                         'supply_added_INT2': 0,
                         'supply_added_INT5': 0,
                         'supply_added_INT6': 0},
        'capacity_added': 0,
        'transadded': 0,
        'referadded': 0,
        'ultracovhome': 0,
        'diagnosisrate': 0,
        'refer_capacity': 100
    }
    return global_vars

def reset_INTs ():
    INTs = {'flag_CHV': 0, 'flag_ANC': 0, 'flag_refer': 0, 'flag_trans': 0,
            'flag_int1': 0, 'flag_int2': 0,'flag_int3': 0, 'flag_int4': 0, 'flag_int5': 0,
            'flag_int6': 0, 'flag_sdr': 0, 'flag_capacity': 0, 'flag_community': 0, 'flag_refer_capacity': 0}
    return INTs

### Side bar ###
with st.sidebar:
    st.header("Outcomes sidebar")
    level_options = ("County", "Subcounty"
                     #, "Subcounty in Map"
                     )
    stop_time = st.slider("When to stop the implementation?",  min_value=1, max_value=35, step=1, value=36)
    select_level = st.selectbox('Select level of interest:', level_options)
    if select_level == "County":
        plotA_options = ("Pathways","Live births",
                         "Maternal deaths", "MMR",
                         "Complications", "Complication rate",
                         "Neonatal deaths", "NMR",
                         "Cost effectiveness",
                         "ANC rate",
                         "Capacity ratio",
                         "Intervention coverage",
                         "Referral from home to L45",
                         "Emergency transfer",
                         "Referral capacity ratio"
                         )
        selected_plotA = st.selectbox(
            label="Select outcome of interest:",
            options=plotA_options,
        )
    if select_level == "Subcounty":
        plotB_options = ("Live births",
                         "Maternal deaths",
                         "MMR",
                         "Complications",
                         "Complication rate",
                         )
        selected_plotB = st.selectbox(
            label="Select outcome of interest:",
            options=plotB_options,
        )
    if select_level == "Subcounty in Map":
        plotC_options = (#"MMR",
                         # "% deliveries in L4/5"
        )
        selected_plotC = st.selectbox(
            label="Choose your interested outcomes in map:",
            options=plotC_options,
        )
        month = st.slider("Which month to show the outcome?",  min_value=1, max_value=35, step=1, value=35)

    select_SCs1 = st.selectbox('Implement SDR policy to all subcounties or part of them?', ['All subcounties', 'Part of subcounties'])
    if select_SCs1 == 'All subcounties':
        select_SCs = subcounties
        SCID_selected = [key for key, value in SC_ID.items() if value in select_SCs]
        SDR_subcounties = [1] * 12
    else:
        SC_options = subcounties
        select_SCs = st.multiselect('Select subcounties:', SC_options)
        SCID_selected = [key for key, value in SC_ID.items() if value in select_SCs]
        SDR_subcounties = [1 if i in SCID_selected else 0 for i in range(12)]

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("SDR (Demand)")
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        st.text('Apply interventions')
        CHVint = st.checkbox('Employ CHVs at communities')
    with col1_2:
        st.text('Adjust parameters')
        if CHVint:
            flag_sdr = 1
            flag_CHV = 1
            CHV_cov = st.slider("CHV Coverage", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
            CHV_45 = st.slider("CHV effect on delivery at L4/5", min_value=0.00, max_value=0.10, step=0.01, value=0.04)
            CHV_pushback = True
            flag_ANC = 1
            ANCadded = st.slider('CHV effect on 4+ANCs', min_value=0.0, max_value=1.2, step=0.1, value=0.2)

with col2:
    st.subheader("SDR (Supply)")
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        st.text('Apply interventions')
        facint = st.checkbox('Upgrade L4/5 facilities')
    with col2_2:
        st.text('Adjust parameters')
        if facint:
            flag_sdr = 1
            know_added = st.slider("Improve knowledge of healthcare workers", min_value=0.0, max_value=1.0,
                                   step=0.1, value=0.2)
            capacity_added = st.slider("Improve facility capacity (labor, equipment, infrastructure)", min_value=0.0, max_value=1.0, step=0.1,
                                       value=0.5)
            supply_added = st.slider("Improve inventory of all single interventions", min_value=0, max_value=20, step=1,
                                     value=1)
            supply_added_INT1 = supply_added
            supply_added_INT2 = supply_added
            supply_added_INT5 = supply_added
            supply_added_INT6 = supply_added
    st.markdown("---")
    col2_3, col2_4 = st.columns(2)
    with col2_3:
        refint = st.checkbox('Rescue network upgrade')
    with col2_4:
        if refint:
            flag_sdr = 1
            refer_capacity = st.slider('Improve referral capacity (per subcounty per month)', min_value=100, max_value=300,
                                       step=1, value=200)

with col3:
    st.subheader("Single interventions")
    col3_1, col3_2 = st.columns(2)
    with col3_1:
        st.text('Apply interventions')
        int1 = st.checkbox('INT1: Obstetric drape for pph')
    with col3_2:
        st.text('Adjust parameters')
        if int1:
            flag_int1 = 1
            value = st.slider("Increase inventory of INT1", min_value=0.0, max_value=0.3, step=0.1,
                                     value=0.1)
            if flag_sdr == 0:
                supply_added_INT1 = value
            else:
                supply_added_INT1 = supply_added_INT1

    st.markdown("---")
    col3_3, col3_4 = st.columns(2)
    with col3_3:
        int2 = st.checkbox('INT2: IV iron infusion for anemia')
    with col3_4:
        if int2:
            flag_int2 = 1
            value = st.slider("Increase inventory of INT2", min_value=0.0, max_value=5.0, step=0.1,
                                          value=0.2)
            if flag_sdr == 0:
                supply_added_INT2 = value
            else:
                supply_added_INT2 = supply_added_INT2
    st.markdown("---")
    col3_5, col3_6 = st.columns(2)
    with col3_5:
        int5 = st.checkbox('INT3: MgSO4 for eclampsia')
    with col3_6:
        if int5:
            flag_int5 = 1
            value = st.slider("Increase inventory of INT5", min_value=0.0, max_value=0.1, step=0.1,
                                          value=0.1)
            if flag_sdr == 0:
                supply_added_INT5 = value
            else:
                supply_added_INT5 = supply_added_INT5
    st.markdown("---")
    col3_7, col3_8 = st.columns(2)
    with col3_7:
        int6 = st.checkbox('INT4: Antibiotics for maternal sepsis (Inventory is already 100%)')
    with col3_8:
        if int6:
            flag_int6 = 1
            if flag_sdr == 0:
                supply_added_INT6 = 0
            else:
                supply_added_INT6 = supply_added_INT6

with ((st.form('Test'))):
    ### PARAMETERs ###
    shapefile_path2 = 'ke_subcounty.shp'

    ### parameters
    def get_parameters():
        param = {
            'pre_comp_home': 0.015,
            'pre_comp_l23': 0.017,
            'pre_comp_l4': 0.031,
            'pre_comp_l5': 0.128,
            'complication_rate': 0.032,
            'p_comp_severe': 0.216,
            'p_comp_ol': 0.01,
            'p_comp_other': 0.0044,
            'p_comp_anemia': 0.25,
            'p_comp_pph': 0.017,
            'p_comp_sepsis': 0.00159,
            'p_comp_eclampsia': 0.0046,
            'or_anemia_pph': 3.54,
            'or_anemia_sepsis': 5.68,
            'or_anemia_eclampsia': 3.74,
            'or_anc_anemia': 2.26,
            't_home_l23': 0,
            't_home_l4': 24.7,
            't_home_l5': 10.335,
            't_l23_l4': 49.36,
            't_l23_l5': 24.422,
            't_l4_l4': 0,
            't_l4_l5': 21.684,
            'm_l_home': 0.005,
            'm_l_l23': 0.004,
            'm_l_l4': 0.004,
            'm_l_l5': 0.003,
            'm_h_home': 25,
            'm_h_l23': 5.18,
            'm_h_l4': 5,
            'm_h_l5': 4.93,
            'm_t': 3.35,
            'i_int1': 0.6,
            'i_int2': 0.7,
            'i_int5': 0.59,
            'i_int6': 0.8,
            'i_int1_supplies': 0.78,
            'i_int2_supplies': 0.17,
            'i_int5_supplies': 0.959,
            'i_int6_supplies': 1,
            'nd_ratio': np.array([28, 28, 19.2, 2])
        }
        return param

    n_months = 36
    t = np.arange(n_months)

    Capacity = np.array(
        [3000.0, 1992.0, 1608.0, 2328.0, 2784.0, 6060.0, 4176.0, 2736.0, 1308.0, 5820.0, 2088.0, 5124.0]) / 12

    param = get_parameters()

    sc_ID = {
        "Butere": 0,
        "Ikolomani": 1,
        "Khwisero": 2,
        "Likuyani": 3,
        "Lugari": 4,
        "Lurambi": 5,
        "Malava": 6,
        "Matungu": 7,
        "Mumias East": 8,
        "Mumias West": 9,
        "Navakholo": 10,
        "Shinyalu": 11
    }

    sc_intervention = {
        'Community Health Workers': 0,
        'Antenatal Care': 1,
        'Antenatal Care Time': 2,
        'Referral': 3,
        'Transfer': 4,
        'Obstetric Drape': 5,
        'Antenatal Corticosteroids': 6,
        'Magnesium Sulfate': 8,
        'Antibiotics for Sepsis': 9,
        'SDR': 10,
        'Capacity': 11,
        'Community': 12
    }

    # PPH, Sepsis, Eclampsia, Obstructed Labor
    comp_transfer_props = np.array([0.2590361446, 0.1204819277, 0.2108433735, 0.4096385542])

    b_comp_mort_rates = np.array([0.07058355, 0.1855773323, 0.3285932084, 0.02096039958, 0.3942855098])


    def reset_inputs(param):
        sc_LB = [
            [1906, 1675, 2245, 118],
            [1799, 674, 1579, 91],
            [1711, 814, 1288, 77],
            [864, 1980, 2045, 99],
            [2949, 1407, 2248, 134],
            [1343, 1377, 73, 3907],
            [3789, 1201, 3349, 169],
            [2366, 2344, 1326, 122],
            [1760, 1370, 868, 81],
            [564, 1696, 2258, 91],
            [2184, 1853, 1579, 114],
            [2494, 1805, 1851, 124]
        ]
        sc_LB = np.array(sc_LB) / 12
        sc_LB = sc_LB.tolist()

        sc_ANC = [
            0.5279693809,
            0.6154368213,
            0.5634318766,
            0.4936848436,
            0.4622291481,
            0.7421207658,
            0.51401622,
            0.5546849627,
            0.5940181417,
            0.6481883272,
            0.4704188482,
            0.5254622251
        ]

        sc_CLASS = [
            0.767669453,
            0.707382485,
            0.583722902,
            0.353483752,
            0.621879554,
            0.341297702,
            0.78264497,
            0.881471916,
            0.815330894,
            0.583155486,
            0.759252685,
            0.708221685
        ]

        i_supplies = [
            param['i_int1_supplies'],
            param['i_int2_supplies'],
            param['i_int5_supplies'],
            param['i_int6_supplies'],
        ]

        sc_knowledge = np.array([
            62.606250,
            61.533333,
            62.700000,
            58.472727,
            70.480000,
            60.587500,
            61.418182,
            61.560000,
            54.350000,
            48.900000,
            66.257100,
            51.960000
        ]) / 100

        sc = pd.DataFrame({
            'LB': sc_LB,
            'ANC': sc_ANC,
            'CLASS': sc_CLASS,
            'knowledge': sc_knowledge
        })

        i_scenarios = pd.DataFrame({
            'Supplies': i_supplies
        })

        sc_time = {
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
            }
        }

        for i in range(sc_time['n']):
            sc_time['LB1s'][i][0, :] = sc['LB'][i]

        INT = {
            'CHV': {
                'n_months_push_back': 2,
                'b_pushback': 5,
                'sdr_effect': np.zeros((12)),
            },
            'ANC': {
                'effect': np.ones((12))
            },
            'refer': {
                'effect': np.ones((12))
            },
            'transfer': {
                'effect': np.ones((12))
            },
            'int1': {
                'effect': np.ones((12)),
                'coverage': np.ones((12))
            },
            'int2': {
                'effect': np.ones((12)),
                'coverage': np.ones((12))
            },
            'int3': {
                'effect': np.ones((12)),
                'coverage': np.ones((12))
            },
            'int4': {
                'effect': np.ones((12)),
                'coverage': np.ones((12))
            },
            'int5': {
                'effect': np.ones((12)),
                'coverage': np.ones((12))
            },
            'int6': {
                'effect': np.zeros((12)),
                'coverage': np.ones((12))
            },
            'SDR': {
                'capacity': np.ones((12)),
                'quality': np.ones((12)),
                'refer_capacity': np.ones((12))
            },
            'Community': {
                'effect': np.ones((12))
            },
            'Capacity': {
                'effect': np.ones((12))
            }
        }

        return INT, sc, i_scenarios, sc_time


    p_ol = param['p_comp_ol']
    p_other = param['p_comp_other']
    p_severe = param['p_comp_severe']

    ### global functions
    def odds_prob(oddsratio, p_comp, p_expose):
        def equations(vars):
            x, y = vars
            eq1 = x / (1 - x) / (y / (1 - y)) - oddsratio
            eq2 = p_comp - p_expose * x - (1 - p_expose) * y
            return [eq1, eq2]

        initial_guess = [0.5, 0.5]  # Initial guess for x and y
        solution = fsolve(equations, initial_guess)
        return solution


    def get_aggregate(df):

        df_aggregate = pd.DataFrame(index=range(n_months), columns=df.columns)
        for i in range(n_months):
            for j in df.columns:
                df_aggregate.loc[i, j] = np.sum(np.array(df.loc[(slice(None), i), j]), axis=0)

        return df_aggregate


    def agg_years(df_agg, n_months):
        df = pd.DataFrame(index=range(round(np.floor(n_months / 12))), columns=df_agg.columns)
        for col in df_agg.columns:
            test = []
            for i in range(1, len(df_agg), 12):
                group = df_agg.iloc[i:i + 12][col].sum()
                test.append(group)
            df[col] = test

        return df

    def community_effect(x):
        if x - 0.5 < 0:
            x = -(1 - x)
        else:
            x = x
        return 0.000015 * (np.exp(10 * x) - np.exp(-10 * x))


    def set_time_int(INTs, time_comp):

        f_intvs = {'flag_CHV': 0, 'flag_ANC': 0, 'flag_refer': 0, 'flag_trans': 0, 'flag_int1': 0, 'flag_int2': 0,
                   'flag_int3': 0, 'flag_int4': 0, 'flag_int5': 0, 'flag_int6': 0, 'flag_sdr': 0, 'flag_capacity': 0,
                   'flag_community': 0, 'flag_refer_capacity': 0}

        for x in f_intvs.keys():
            if INTs[x] == 1:
                f_intvs[x] = np.array(
                    [0] * time_comp[0] + [1] * (time_comp[1] - time_comp[0]) + [0] * (n_months - time_comp[1]))
            else:
                f_intvs[x] = np.array([0] * n_months)
        return f_intvs

    def set_INT(param, i, j, subcounty, f_intvs, i_scenarios, sc, comps, global_vars):
        flag_CHV = f_intvs['flag_CHV'][i]
        flag_ANC = f_intvs['flag_ANC'][i]
        flag_refer = f_intvs['flag_refer'][i]
        flag_trans = f_intvs['flag_trans'][i]
        flag_int1 = f_intvs['flag_int1'][i]
        flag_int2 = f_intvs['flag_int2'][i]
        flag_int5 = f_intvs['flag_int5'][i]
        flag_int6 = f_intvs['flag_int6'][i]
        flag_sdr = f_intvs['flag_sdr'][i]
        flag_capacity = f_intvs['flag_capacity'][i]
        flag_community = f_intvs['flag_community'][i]
        flag_refer_capacity = f_intvs['flag_refer_capacity'][i]

        Capacity = np.array(
            [3000.0, 1992.0, 1608.0, 2328.0, 2784.0, 6060.0, 4176.0, 2736.0, 1308.0, 5820.0, 2088.0, 5124.0]) / 12
        base_comp = np.array(
            [0.5781658344485783, 0.5746317314613323, 0.5242427518609455, 0.6190109739999066, 0.5270073028715008,
             0.8267657647642364, 0.5823162795099026, 0.40824676781240365, 0.401120698300252, 0.6891886648218939,
             0.4737322018498702, 0.49205803086845007])

        INT, sc, i_scenarios, _ = reset_inputs(param)

        supplies = i_scenarios['Supplies']
        knowl = sc['knowledge'][j]
        weight = 0.448  # (LB_tot[2] + LB_tot[3])/(np.sum(LB_tot))
        b_usage = [sc['ANC'][j], base_comp[j], base_comp[j], base_comp[j]]
        i_usage = [sc['ANC'][j] * INT['ANC']['effect'][j], comps, comps, comps]  #here INT['ANC']['effect'][j] = 1?
        comps_ratio = comps / base_comp[j]

        suppl = np.minimum(np.array([s for s in supplies]) / comps_ratio, 1)
        base = np.minimum(suppl * knowl * b_usage, 1)

        supply_added_INT1 = global_vars['supply_added']['supply_added_INT1']
        supply_added_INT2 = global_vars['supply_added']['supply_added_INT2']
        supply_added_INT5 = global_vars['supply_added']['supply_added_INT5']
        supply_added_INT6 = global_vars['supply_added']['supply_added_INT6']
        know_added = global_vars['know_added']
        capacity_added = global_vars['capacity_added']
        CHV_45 = global_vars['CHV_45']
        CHV_cov = global_vars['CHV_cov']
        ANCadded = global_vars['ANCadded']
        transadded = global_vars['transadded']
        referadded = global_vars['referadded']

        # usage
        if subcounty[j] == 1:
            if flag_sdr:
                knowl = min(1, (1 + know_added) * (knowl))
                #int_suppl = np.minimum(((1 + supply_added) * suppl) / comps_ratio, 1)
                flag_CHV = 1
                flag_int1 = 1
                flag_int2 = 1
                flag_int5 = 1
                flag_int6 = 1
                flag_capacity = 1
                flag_refer_capacity = 1
            if flag_CHV:
                INT['CHV']['sdr_effect'][j] = CHV_45 * CHV_cov
                flag_ANC = 1
            else:
                INT['CHV']['sdr_effect'][j] = 0
            if flag_ANC:
                INT['ANC']['effect'][j] = 1 + ANCadded * CHV_cov / n_months * i
                max_anc = min(1, sc['ANC'][j] * INT['ANC']['effect'][j])
                INT['ANC']['effect'][j] = max_anc / sc['ANC'][j]
                i_usage = [sc['ANC'][j] * INT['ANC']['effect'][j], comps, comps, comps]
            else:
                INT['ANC']['effect'][j] = 1
            if flag_refer:
                # referral intervention: increases self-referrals to higher level facilities, considered as pre measurement
                INT['refer']['effect'][j] = 1 + referadded
            else:
                INT['refer']['effect'][j] = 1
            if flag_trans:
                # transfer intervention: increases transfer rate of complications
                INT['transfer']['effect'][j] = 1 + transadded
            else:
                INT['transfer']['effect'][j] = 1
            if flag_int1:
                int_suppl = np.minimum(((1 + supply_added_INT1) * suppl) / comps_ratio, 1)
                INT['int1']['coverage'][j] = min(knowl * int_suppl[0] * i_usage[0], 1)
                quality_factor = INT['int1']['coverage'][j] - base[0]
                INT['int1']['effect'][j] = 1 - ((1 - param['i_int1']) * quality_factor)
                # facility
            else:
                quality_factor = base[0]
                INT['int1']['coverage'][j] = quality_factor
            if flag_int2:
                int_suppl = np.minimum(((1 + supply_added_INT2) * suppl) / comps_ratio, 1)
                # INT['int2']['coverage'][j] = min(knowl * int_suppl * i_usage[1], 1)
                INT['int2']['coverage'][j] = min(knowl * int_suppl[1] * i_usage[1], 1)  # quality_factor
                quality_factor = INT['int2']['coverage'][j] - base[1]
                INT['int2']['effect'][j] = 1 - ((1 - param['i_int2']) * quality_factor)
            else:
                quality_factor = base[1]
                INT['int2']['coverage'][j] = quality_factor
            if flag_int5:
                int_suppl = np.minimum(((1 + supply_added_INT5) * suppl) / comps_ratio, 1)
                INT['int5']['coverage'][j] = min(knowl * int_suppl[2] * i_usage[2], 1)  # quality_factor
                quality_factor = INT['int5']['coverage'][j] - base[2]
                INT['int5']['effect'][j] = 1 - ((1 - param['i_int5']) * quality_factor)
            else:
                quality_factor = base[2]
                INT['int5']['coverage'][j] = quality_factor
            if flag_int6:
                int_suppl = np.minimum(((1 + supply_added_INT6) * suppl) / comps_ratio, 1)
                INT['int6']['coverage'][j] = min(knowl * int_suppl[3] * i_usage[3], 1)  # quality_factor
                quality_factor = INT['int6']['coverage'][j] - base[3]
                INT['int6']['effect'][j] = 1 - ((1 - param['i_int6']) * quality_factor)
            else:
                quality_factor = base[3]
                INT['int6']['coverage'][j] = quality_factor
            if flag_capacity:
                Capacity[j] = Capacity[j] * (1 + capacity_added / n_months * i)
            else:
                Capacity[j] = Capacity[j]
            if flag_community:
                INT['Community']['effect'][j] = 1
            else:
                INT['Community']['effect'][j] = 0
            if flag_refer_capacity:
                INT['SDR']['refer_capacity'][j] = global_vars['refer_capacity']
            else:
                INT['SDR']['refer_capacity'][j] = 100

        else:
            INT['CHV']['sdr_effect'][j] = 0
            INT['ANC']['effect'][j] = 1
            INT['refer']['effect'][j] = 1
            INT['transfer']['effect'][j] = 1
            INT['int1']['coverage'][j] = base[0]
            INT['int2']['coverage'][j] = base[1]
            INT['int5']['coverage'][j] = base[2]
            INT['int6']['coverage'][j] = base[3]
            INT['int1']['effect'][j] = 1
            INT['int2']['effect'][j] = 1
            INT['int5']['effect'][j] = 1
            INT['int6']['effect'][j] = 1
            Capacity[j] = Capacity[j]
            INT['Community']['effect'][j] = 0
            INT['SDR']['refer_capacity'][j] = 100
        return INT, Capacity

    def get_cost_effectiveness(INTs, timepoint, df_years, b_df_years, global_vars):
        DALYs = {'low pph': 0.114,
                 'high pph': 0.324,
                 'sepsis': 0.133,
                 'eclampsia': 0.602,
                 'obstructed labor': 0.324,
                 'death': 0.54,
                 'neonatal death': 1
        }


        flag_sdr = INTs['flag_sdr']  # SDR
        if flag_sdr:
            flag_int1 = 1  # Obstetric drape
            flag_int2 = 1  # Anemia reduction through IV Iron
            flag_int5 = 1  # MgSO4 for eclampsia
            flag_int6 = 1  # antibiotics for maternal sepsis
        else:
            flag_int1 = INTs['flag_int1']  # Obstetric drape
            flag_int2 = INTs['flag_int2']  # Anemia reduction through IV Iron
            flag_int5 = INTs['flag_int5']  # MgSO4 for eclampsia
            flag_int6 = INTs['flag_int6']  # antibiotics for maternal sepsis

        DALYs_averted = 0
        b_low_pph_all = 0
        b_high_pph_all = 0
        b_eclampsia_all = 0
        b_sepsis_all = 0

        anc_cost = 0
        delivery_cost = 0
        c_section_cost = 0
        labor_cost = 0
        equipment_cost = 0
        intervention_cost = 0
        infrastructure_cost = 0
        chv_cost = 0
        access_cost = 0

        addfac_deliveries = 0
        addanc = 0
        b_anc_all = 0
        pctaddanc = 0
        pctaddfac_deliveries = 0
        addreferral = 0

        low_pph_all = 0
        high_pph_all = 0
        pph_all = 0
        eclampsia_all = 0
        sepsis_all = 0
        obstructed_all = 0
        other_all = 0

        b_int1_covered = 0
        i_int1_covered = 0
        b_int2_covered = 0
        i_int2_covered = 0
        b_int5_covered = 0
        i_int5_covered = 0
        b_int6_covered = 0
        i_int6_covered = 0

        L45deliveries = 0

            #INT1 cover
        for i in range(3):
            #Baseline
            low_pph1 = np.sum(b_df_years.loc[i, 'Complications-Health'], axis=1)[0] * \
                      np.array([1 - p_severe, p_severe])[0]
            high_pph1 = np.sum(b_df_years.loc[i, 'Complications-Health'], axis=1)[0] * \
                       np.array([1 - p_severe, p_severe])[1]
            sepsis1 = np.sum(b_df_years.loc[i, 'Complications-Health'], axis=1)[1]
            eclampsia1 = np.sum(b_df_years.loc[i, 'Complications-Health'], axis=1)[2]
            obstructed1 = np.sum(b_df_years.loc[i, 'Complications-Health'], axis=1)[3]
            maternal_deaths1 = np.sum(b_df_years.loc[i, 'Deaths'])
            neonatal_deaths1 = np.sum(b_df_years.loc[i, 'Neonatal Deaths'])
            b_DALYs1 = (DALYs['low pph'] * low_pph1 + DALYs['high pph'] * high_pph1
                        + DALYs['sepsis'] * sepsis1 + DALYs['eclampsia'] * eclampsia1 + DALYs['obstructed labor'] * obstructed1
                        + DALYs['death'] * maternal_deaths1)* 62.63 + DALYs['neonatal death'] * neonatal_deaths1 * 60.25
            b_low_pph_all += low_pph1
            b_high_pph_all += high_pph1
            b_eclampsia_all += eclampsia1
            b_sepsis_all += sepsis1

            #Intervention
            low_pph1 = np.sum(df_years.loc[i, 'Complications-Health'], axis=1)[0] * \
                      np.array([1 - p_severe, p_severe])[0]
            high_pph1 = np.sum(df_years.loc[i, 'Complications-Health'], axis=1)[0] * \
                       np.array([1 - p_severe, p_severe])[1]
            sepsis1 = np.sum(df_years.loc[i, 'Complications-Health'], axis=1)[1]
            eclampsia1 = np.sum(df_years.loc[i, 'Complications-Health'], axis=1)[2]
            obstructed1 = np.sum(df_years.loc[i, 'Complications-Health'], axis=1)[3]
            other1 = np.sum(df_years.loc[i, 'Complications-Health'], axis=1)[4]
            maternal_deaths1 = np.sum(df_years.loc[i, 'Deaths'])
            neonatal_deaths1 = np.sum(df_years.loc[i, 'Neonatal Deaths'])
            i_DALYs1 = (DALYs['low pph'] * low_pph1 + DALYs['high pph'] * high_pph1
                        + DALYs['sepsis'] * sepsis1 + DALYs['eclampsia'] * eclampsia1 + DALYs['obstructed labor'] * obstructed1
                        + DALYs['death'] * maternal_deaths1) * 62.63 + DALYs['neonatal death'] * neonatal_deaths1 * 60.25
            low_pph_all += low_pph1
            high_pph_all += high_pph1
            pph_all = low_pph_all + high_pph_all
            eclampsia_all += eclampsia1
            sepsis_all += sepsis1
            obstructed_all += obstructed1
            other_all += other1

            DALYs_averted1 = b_DALYs1 - i_DALYs1
            DALYs_averted += DALYs_averted1

            b_fac_deliveries = np.sum(b_df_years.loc[i, 'Live Births Final']) - b_df_years.loc[i, 'Live Births Final'][0]
            fac_deliveries = np.sum(df_years.loc[i, 'Live Births Final']) - df_years.loc[i, 'Live Births Final'][0]
            addfac_deliveries1 = fac_deliveries - b_fac_deliveries
            addfac_deliveries += addfac_deliveries1
            addCsections = addfac_deliveries * 0.098
            total_LB = np.sum(df_years.loc[i, 'Live Births Final'])
            pctaddfac_deliveries = addfac_deliveries * 100 / (total_LB * 3)

            b_referral = np.sum(b_df_years.loc[i, 'Normal Referral'] + b_df_years.loc[i, 'Complication Referral'])
            referral = np.sum(df_years.loc[i, 'Normal Referral'] + df_years.loc[i, 'Complication Referral'])
            addreferral1 = referral - b_referral
            addreferral += addreferral1

            b_anc = b_df_years.loc[i, 'LB-ANC'][1]
            b_anc_all += b_anc
            anc = df_years.loc[i, 'LB-ANC'][1]
            addanc1 = anc - b_anc
            addanc += addanc1
            pctaddanc = addanc * 100 / (total_LB * 3)

            if i == 0:
                addfac_deliveries_year1 = addfac_deliveries
                addCsections_year1 = addCsections
                addanc_year1 = addanc
            if i == 1:
                addfac_deliveries_year2 = addfac_deliveries
                addCsections_year2 = addCsections
                addanc_year2 = addanc
            if i == 2:
                addfac_deliveries_year3 = addfac_deliveries
                addCsections_year3 = addCsections
                addanc_year3 = addanc

            ##INT coverage
            int1_covered = b_df_years.loc[i, 'Mothers with INT1']
            b_int1_covered += int1_covered
            int1_covered = df_years.loc[i, 'Mothers with INT1']
            i_int1_covered += int1_covered

            int2_covered = b_df_years.loc[i, 'Mothers with INT2']
            b_int2_covered += int2_covered
            int2_covered = df_years.loc[i, 'Mothers with INT2']
            i_int2_covered += int2_covered

            int5_covered = b_df_years.loc[i, 'Mothers with INT5']
            b_int5_covered += int5_covered
            int5_covered = df_years.loc[i, 'Mothers with INT5']
            i_int5_covered += int5_covered

            int6_covered = b_df_years.loc[i, 'Mothers with INT6']
            b_int6_covered += int6_covered
            int6_covered1 = df_years.loc[i, 'Mothers with INT6']
            i_int6_covered += int6_covered

            #Baseline facility delivery rate
            L45deliveries1 = np.sum(b_df_years.loc[i, 'Live Births Final'][2:4])
            L45deliveries += L45deliveries1
            L45rate = L45deliveries / (total_LB * 3)

        if flag_sdr:
            anc_cost += 288 / 110 * (addanc_year1 * (1 + 0.03) + addanc_year2 * (1 + 0.03) ** 2 + addanc_year3 * (1 + 0.03) ** 3)
            delivery_cost = 6148 / 110 * (addfac_deliveries_year1 * (1 + 0.03) + addfac_deliveries_year2 * (
                        1 + 0.03) ** 2 + addfac_deliveries_year3 * (1 + 0.03) ** 3)
            c_section_cost = 29804 / 110 * (addCsections_year1 * (1 + 0.03) + addCsections_year2 * (1 + 0.03) ** 2 + addCsections_year3 * (1 + 0.03) ** 3)

            equipment_cost = 39984826 / 110 * (global_vars['capacity_added'] * L45rate * 10 / 15)
            labor_cost = 370785580 / 110 * (global_vars['capacity_added'] * L45rate * 10 / 15) * 3 * (timepoint / 36)
            infrastructure_cost = 136600000 / 110 * (global_vars['capacity_added'] * L45rate * 10 / 15)

            chv_cost = (1974 * 420) / 110 * sum(SDR_subcounties) * CHV_cov + (1974 * 2 * 33) / 110 * sum(SDR_subcounties) * CHV_cov * 3 * (timepoint / 36) + 100 / 110 * addfac_deliveries
            #intervention_cost = (1 * (b_high_pph_all + b_high_pph_all) * global_vars['supply_added']['supply_added_INT1'] +
            #                     2.26 * b_anc_all * global_vars['supply_added']['supply_added_INT2'] + 3 * b_eclampsia_all * global_vars['supply_added']['supply_added_INT5'] +
            #                     12.30 * b_sepsis_all * global_vars['supply_added']['supply_added_INT6']) * (timepoint / 36)
            access_cost = 9287 / 110 * addreferral
        #else:
        if flag_int1:
            intervention_cost += 1 * (i_int1_covered - b_int1_covered)
            #intervention_cost += 1 * (b_high_pph_all + b_high_pph_all) * global_vars['supply_added']['supply_added_INT1'] * (timepoint / 36)
        if flag_int2:
            intervention_cost += 2.26 * (i_int2_covered - b_int2_covered)
            #intervention_cost += 2.26 * b_anc_all * global_vars['supply_added']['supply_added_INT2'] * (timepoint / 36)
        if flag_int5:
            intervention_cost += 3 * (i_int5_covered - b_int5_covered)
        #intervention_cost += 3 * b_eclampsia_all * global_vars['supply_added']['supply_added_INT5'] * (timepoint / 36)
        if flag_int6:
            intervention_cost += 12.30 * (i_int6_covered - b_int6_covered)
            #intervention_cost += 12.30 * b_sepsis_all * global_vars['supply_added']['supply_added_INT6'] * (timepoint / 36)

        cost = anc_cost + intervention_cost + equipment_cost + delivery_cost + c_section_cost + labor_cost + infrastructure_cost + chv_cost + access_cost

        return cost, anc_cost, intervention_cost, equipment_cost, delivery_cost, c_section_cost, labor_cost, infrastructure_cost, chv_cost, access_cost, pph_all, eclampsia_all, sepsis_all, obstructed_all, other_all, pctaddanc, pctaddfac_deliveries, DALYs_averted

    def set_param(param):

        f_comps = np.array([param['pre_comp_home'], param['pre_comp_l23'], param['pre_comp_l4'], param['pre_comp_l5']])
        comps_overall = param['complication_rate']
        f_refer = comps_overall - f_comps

        f_transfer_rates = np.array([
            [0.00, param['t_home_l23'], param['t_home_l4'], param['t_home_l5']],
            [0.00, 0.00, param['t_l23_l4'], param['t_l23_l5']],
            [0.00, 0.00, 0, param['t_l4_l5']],
            [0.00, 0.00, 0.00, 0.00]
        ]) / 100

        m_transfer = param['m_t']

        f_mort_rates = np.array([
            [param['m_l_home'], param['m_h_home']],
            [param['m_l_l23'], param['m_h_l23']],
            [param['m_l_l4'], param['m_h_l4']],
            [param['m_l_l5'], param['m_h_l5']]
        ]) / 100
        f_mort_rates = np.c_[f_mort_rates, f_mort_rates[:, 1] * m_transfer]

        return f_comps, comps_overall, f_refer, f_transfer_rates, f_mort_rates


    def run_model(param, subcounty, f_intvs, t, n_months, global_vars):

        i_INT, sc, i_scenarios, sc_time = reset_inputs(param)

        Capacity = np.array(
            [3000.0, 1992.0, 1608.0, 2328.0, 2784.0, 6060.0, 4176.0, 2736.0, 1308.0, 5820.0, 2088.0, 5124.0]) / 12
        b_usage = np.array(
            [0.5781658344485783, 0.5746317314613323, 0.5242427518609455, 0.6190109739999066, 0.5270073028715008,
             0.8267657647642364, 0.5823162795099026, 0.40824676781240365, 0.401120698300252, 0.6891886648218939,
             0.4737322018498702, 0.49205803086845007])

        Capacity_ratio = np.ones((n_months, sc_time['n']))
        Capacity_ratio_ref = np.ones((sc_time['n'], n_months))
        Push_back = np.zeros((n_months, sc_time['n']))
        opinion = np.zeros((n_months, sc_time['n']))
        normal_ref = np.zeros((sc_time['n'], n_months))
        com_ref = np.zeros((sc_time['n'], n_months))
        comps = np.zeros((sc_time['n'], n_months))
        comps[:, 0] = b_usage

        f_comps, f_transfer_rates, f_mort_rates = set_param(param)

        p_anemia = param['p_comp_anemia']
        p_pph = param['p_comp_pph']
        p_sepsis = param['p_comp_sepsis']
        p_eclampsia = param['p_comp_eclampsia']

        flag_pushback = global_vars['CHV_pushback']
        Push_back = np.zeros((n_months, sc_time['n']))

        index = pd.MultiIndex.from_product([range(sc_time['n']), range(n_months)], names=['Subcounty', 'Time'])

        df_3d = pd.DataFrame(index=index, columns=['Live Births Initial', 'Live Births Final', 'LB-ANC', 'ANC-Anemia',
                                                   'Anemia-Complications',
                                                   'Complications-Facility Level', 'Facility Level-Complications Pre',
                                                   'Transferred In', 'Transferred Out',
                                                   'Facility Level-Complications Post', 'Facility Level-Severity',
                                                   'Transfer Severity Mortality', 'Deaths', 'Facility Level-Health',
                                                   'Complications-Survive', 'Complications-Health', 'Complications-All',
                                                   'Complications-Complications',
                                                   'Capacity Ratio', 'Push Back', 'INT1 cover', 'INT2 cover',
                                                   'INT5 cover', 'INT6 cover',
                                                   'Normal Referral', 'Complication Referral',
                                                   'Referral capacity ratio', 'Mothers with INT1', 'Mothers with INT2', 'Mothers with INT5', 'Mothers with INT6','Neonatal Deaths'])

        for i in t:
            LB_tot_i = np.zeros(4)
            for j in range(sc_time['n']):

                if i > 0:
                    i_INT, Capacity = set_INT(param, i, j, subcounty, f_intvs, i_scenarios, sc, comps[j, i - 1],
                                              global_vars)

                    probs = []
                    p_anc_anemia = odds_prob(param['or_anc_anemia'], p_anemia, 1 - sc['ANC'][j]) * (
                                1 - sc['ANC'][j] * i_INT['ANC']['effect'][j]) / (1 - sc['ANC'][j])
                    p_anemia_pph = np.array(odds_prob(param['or_anemia_pph'], p_pph, p_anemia)) * \
                                   i_INT['int2']['effect'][j]
                    p_anemia_sepsis = np.array(odds_prob(param['or_anemia_sepsis'], p_sepsis, p_anemia)) * \
                                      i_INT['int2']['effect'][j]
                    p_anemia_eclampsia = np.array(odds_prob(param['or_anemia_eclampsia'], p_eclampsia, p_anemia)) * \
                                         i_INT['int2']['effect'][j]
                    probs.extend([p_anc_anemia, p_anemia_pph, p_anemia_sepsis, p_anemia_eclampsia])

                    #st.text(i_INT['int2']['effect'][j])


                    sc_time['LB1s'][j][i, :], Push_back[i, j] = f_LB_effect(sc_time['LB1s'][j][0, :], com_ref[j, i - 1],
                                                                            sc_time['LB1s'][j][i - 1, :], i_INT,
                                                                            flag_pushback, j, i, Capacity_ratio[:, j],
                                                                            opinion[i, j], Capacity)
                    LB_tot_i = np.maximum(sc_time['LB1s'][j][0, :], 1)
                    SDR_multiplier = sc_time['LB1s'][j][i, :] / LB_tot_i

                    anc = sc['ANC'][j] * i_INT['ANC']['effect'][j]
                    comps_overall = np.sum((f_comps * sc_time['LB1s'][j][i, :])) / np.sum(sc_time['LB1s'][j][i, :])
                    f_refer = comps_overall - f_comps

                    LB_tot, LB_tot_final, lb_anc, anc_anemia, anemia_comps_pre, comps_level_pre, f_comps_level_pre, transferred_in, transferred_out, f_comps_level_post, f_severe_post, f_state_mort, f_deaths, f_health, f_comp_survive, comps_health, comps_all, comps_comps = f_MM(
                        LB_tot_i, anc, i_INT, SDR_multiplier, f_transfer_rates, f_mort_rates, f_refer, comps_overall,
                        probs, sc, j, i)

                    ##sc_time['LB1s'][j][i,:] = LB_tot = LB before complication transfer, LB_tot_final = LB after complication transfer
                    ##We can assume complication transfer occupy the rescue quota of next month
                    complication_ref = np.sum(transferred_in + transferred_out)
                    base_normal_ref = complication_ref * 0.18
                    add_normal_ref = sum((sc_time['LB1s'][j][i, :] - sc_time['LB1s'][j][0, :])[
                                         2:4])
                    normal_ref[j, i] = max(base_normal_ref, add_normal_ref)
                    com_ref[j, i] = complication_ref
                    Capacity_ratio_ref[j, i] = np.sum(normal_ref[j, i] + com_ref[j, i]) / i_INT['SDR']['refer_capacity'][j]

                    df_3d.loc[(j, i), ['Live Births Initial', 'Live Births Final', 'LB-ANC', 'ANC-Anemia',
                                       'Anemia-Complications', 'Complications-Facility Level',
                                       'Facility Level-Complications Pre', 'Transferred In', 'Transferred Out',
                                       'Facility Level-Complications Post', 'Facility Level-Severity',
                                       'Transfer Severity Mortality', 'Deaths', 'Facility Level-Health',
                                       'Complications-Survive', 'Complications-Health', 'Complications-All', 'Complications-Complications',
                                       'Push Back', 'INT1 cover', 'INT2 cover', 'INT5 cover', 'INT6 cover',
                                       'Normal Referral', 'Complication Referral', 'Referral capacity ratio']] = [
                        LB_tot, LB_tot_final.astype(float), np.round(lb_anc, decimals=2), anc_anemia, anemia_comps_pre,
                        comps_level_pre, f_comps_level_pre, transferred_in.astype(float), transferred_out.astype(float),
                        f_comps_level_post, f_severe_post, f_state_mort, f_deaths, f_health, f_comp_survive,
                        np.round(comps_health, decimals=3), comps_all, comps_comps, Push_back[i, j], i_INT['int1']['coverage'][j],
                        i_INT['int2']['coverage'][j], i_INT['int5']['coverage'][j], i_INT['int6']['coverage'][j],
                        normal_ref[j, i], com_ref[j, i], Capacity_ratio_ref[j, i]]

                    comps[j, i] = np.sum(f_comps_level_post[2:4]) / (np.sum(f_comps_level_post))

                    #pph covered
                    df_3d.loc[(j, i), 'Mothers with INT1'] = comps_all[0] * i_INT['int1']['coverage'][j]
                    #anemia covered
                    df_3d.loc[(j, i), 'Mothers with INT2'] = np.sum(LB_tot) * i_INT['int2']['coverage'][j]
                    #eclampisa covered
                    df_3d.loc[(j, i), 'Mothers with INT5'] = comps_all[2] * i_INT['int5']['coverage'][j]
                    #sepsis covered
                    df_3d.loc[(j, i), 'Mothers with INT6'] = comps_all[1] * i_INT['int6']['coverage'][j]
                    #df_3d.loc[(j, i), 'Neonatal Deaths'] = np.sum(comps_health, axis=0)[0] * param['nd_ratio']
                    df_3d.loc[(j, i), 'Neonatal Deaths'] = f_deaths * param['nd_ratio']

                if j != 5:
                    Capacity_ratio[i, j] = np.sum(sc_time['LB1s'][j][i, 2:3]) / Capacity[j]
                    opinion[i, j] = np.sum(sc_time['LB1s'][j][i, 2:3]) / np.sum(sc_time['LB1s'][j][i, :3])
                else:
                    Capacity_ratio[i, j] = np.sum(sc_time['LB1s'][j][i, 3:4]) / Capacity[j]
                    opinion[i, j] = np.sum(sc_time['LB1s'][j][i, 3:4]) / np.sum(sc_time['LB1s'][j][i, list([0, 1, 3])])
                df_3d.loc[(j, i), 'Capacity Ratio'] = Capacity_ratio[i, j]
                LB_tot_i += sc_time['LB1s'][j][i, :]

        return df_3d

    def f_LB_effect(SC_LB_base, Com_ref, SC_LB_previous, INT, flag_pushback, j, i, Capacity_ratio, opinion, Capacity):
        referral_capacity = INT['SDR']['refer_capacity'][j]
        INT_b = INT['CHV']['sdr_effect'][j]
        INT_c = INT['Community']['effect'][j] * community_effect(opinion) * (INT_b > 0)
        if flag_pushback:
            i_months = range(max(0, i - INT['CHV']['n_months_push_back']), i)
            mean_capacity_ratio = np.mean(Capacity_ratio[i_months])
            push_back = max(0, mean_capacity_ratio - 1)
            scale = np.exp(-push_back * INT['CHV']['b_pushback'])
            INT_b = INT_b * scale + INT_c
            effect = np.array([np.exp(-INT_b * np.array([1, 1])), 1 + INT_b * np.array([1, 1])]).reshape(1,
                                                                                                         4)  # effect of intervention on LB
            SC_LB = SC_LB_previous * effect  # new numbers of LB
            SC_LB = SC_LB / np.sum(SC_LB) * np.sum(SC_LB_previous)  # sum should equal the prior sum
            ## this accounts for referral capacity ##
            if INT_b > 0:
                exp_normal_ref = np.sum((SC_LB - SC_LB_base)[0][2:4])
                max_normal_refer = min(exp_normal_ref, referral_capacity - np.sum(Com_ref))
                change = max_normal_refer / exp_normal_ref * (SC_LB - SC_LB_base)
                SC_LB = SC_LB_base + change
            ## this accounts for facility overcapacity ##
            overcapacity = np.sum(SC_LB[0][2:4]) / Capacity[j]
            if overcapacity >= 1.2:
                SC_LB[0][2] = SC_LB[0][2] - (overcapacity - 1.2) * Capacity[j]
                SC_LB[0][1] = SC_LB[0][1] + (overcapacity - 1.2) * Capacity[j]
            return SC_LB, push_back
        else:
            INT_b = INT_b + INT_c
            effect = np.array([np.exp(-INT_b * np.array([1, 1])), 1 + INT_b * np.array([1, 1])]).reshape(1,
                                                                                                         4)  # effect of intervention on LB
            SC_LB = SC_LB_previous * effect  # new numbers of LB
            SC_LB = SC_LB / np.sum(SC_LB) * np.sum(SC_LB_previous)  # sum should equal the prior sum
            return SC_LB, 0


    def f_MM(LB_tot, anc, INT, SDR_multiplier_0, f_transfer_rates, f_mort_rates, f_refer, comps_overall, probs, sc, j,
             i):

        LB_tot = np.array(LB_tot) * SDR_multiplier_0

        p_anc_anemia = probs[0]
        p_anemia_pph = probs[1]
        p_anemia_sepsis = probs[2]
        p_anemia_eclampsia = probs[3]

        original_comps = LB_tot * comps_overall
        refer = (1 - sc['CLASS'][j]) + INT['refer']['effect'][j] * (sc['CLASS'][j])
        new_comps = original_comps - LB_tot * f_refer * refer
        f_comps_adj = new_comps / (LB_tot + new_comps - original_comps)

        lb_anc = np.array([1 - anc, anc]) * np.sum(LB_tot)  ### lb_anc
        anc_anemia = np.array([
            [lb_anc[0] * (1 - p_anc_anemia[0]), lb_anc[1] * (1 - p_anc_anemia[1])],
            [p_anc_anemia[0] * lb_anc[0], p_anc_anemia[1] * lb_anc[1]]
        ])  ### anc_anemia

        anemia_comps = np.array([
            [p_anemia_pph[1], p_anemia_pph[0]],
            [p_anemia_sepsis[1], p_anemia_sepsis[0]],
            [p_anemia_eclampsia[1], p_anemia_eclampsia[0]],
            [0.5 * p_ol, 0.5 * p_ol],
            [0.5 * p_other, 0.5 * p_other]
        ])

        anemia_comps_pre = anemia_comps * np.sum(anc_anemia,
                                                 axis=1)  ### complications by no anemia, anemia - anemia_comp
        f_comps_prop = (LB_tot * f_comps_adj) / np.sum(LB_tot * f_comps_adj)
        comp_reduction = np.ones((5, 4))
        comp_reduction[0, 2:4] = INT['int1']['effect'][j]  # PPH
        comp_reduction[2, 2:4] = INT['int5']['effect'][j]  # eclampsia
        comps_level_pre = np.sum(anemia_comps_pre, axis=1)[:,
                          np.newaxis] * f_comps_prop  # complications by facility level
        comps_level_pre = comps_level_pre * comp_reduction
        anemia_comp_props = anemia_comps_pre / (np.sum(anemia_comps_pre, axis=1)[:, np.newaxis])
        change = (np.sum(anemia_comps_pre, axis=1) - np.sum(comps_level_pre, axis=1))[:, np.newaxis] * anemia_comp_props
        anemia_comps_pre = anemia_comps_pre - change
        # anemia_comps_pre

        f_comps_level_pre = np.sum(comps_level_pre, axis=0)
        f_severe_pre = f_comps_level_pre[:, np.newaxis] * np.array([1 - p_severe, p_severe])
        f_transfer_rates = f_transfer_rates * INT['transfer']['effect'][j]
        transferred_in = f_severe_pre[:, 1] @ f_transfer_rates
        transferred_out = np.sum(f_severe_pre[:, 1].reshape(1, 4) * f_transfer_rates.T, axis=0)
        f_comps_level_post = f_comps_level_pre + transferred_in - transferred_out

        f_severe_post = f_comps_level_pre[:, np.newaxis] * np.array([1 - p_severe, p_severe]) + np.array(
            [np.zeros(4), transferred_in]).T - np.array([np.zeros(4), transferred_out]).T
        # ## in case where more transfers than severe complications
        # severe, not transferred, severe transferred, not severe
        f_state_mort = np.c_[f_comps_level_pre[:, np.newaxis] * np.array([1 - p_severe, p_severe]) - np.array(
            [np.zeros(4), transferred_out]).T, transferred_in]
        f_deaths = np.sum(f_state_mort * f_mort_rates, axis=1)
        f_comp_survive = f_state_mort - f_state_mort * f_mort_rates
        f_health = np.c_[f_deaths, np.sum(f_comp_survive, axis=1)]

        temp = sum(np.sum(comps_level_pre, axis=1) * b_comp_mort_rates)
        comp_props = np.sum(comps_level_pre, axis=1) * b_comp_mort_rates / temp
        comps_health = np.c_[
            np.sum(f_deaths) * comp_props, np.sum(comps_level_pre, axis=1) - np.sum(f_deaths) * comp_props]  ###
        mort_reduction = np.zeros(5)
        mort_reduction[1] = INT['int6']['effect'][j]  # sepsis
        mort_reduction = mort_reduction * comps_health[:, 0]  # sepsis
        comps_health[:, 0] = comps_health[:, 0] - mort_reduction
        comps_health[:, 1] = comps_health[:, 1] + mort_reduction
        comps_all = comps_health[:, 0] + comps_health[:, 1]

        comps_comps = f_severe_pre[:, 1].reshape(4, 1) * f_transfer_rates
        for i in range(4):
            comps_comps[i, i] = f_comps_level_pre[i] - np.sum(comps_comps, axis=1)[i]

        LB_tot_final = LB_tot + transferred_in - transferred_out

        return LB_tot, LB_tot_final, lb_anc, anc_anemia, anemia_comps_pre, comps_level_pre, f_comps_level_pre, transferred_in, transferred_out, f_comps_level_post, f_severe_post, f_state_mort, f_deaths, f_health, f_comp_survive, comps_health, comps_all, comps_comps


    def set_param(param):

        f_comps = np.array([param['pre_comp_home'], param['pre_comp_l23'], param['pre_comp_l4'], param['pre_comp_l5']])

        f_transfer_rates = np.array([
            [0.00, param['t_home_l23'], param['t_home_l4'], param['t_home_l5']],
            [0.00, 0.00, param['t_l23_l4'], param['t_l23_l5']],
            [0.00, 0.00, 0, param['t_l4_l5']],
            [0.00, 0.00, 0.00, 0.00]
        ]) / 100

        m_transfer = param['m_t']

        f_mort_rates = np.array([
            [param['m_l_home'], param['m_h_home']],
            [param['m_l_l23'], param['m_h_l23']],
            [param['m_l_l4'], param['m_h_l4']],
            [param['m_l_l5'], param['m_h_l5']]
        ]) / 100
        f_mort_rates = np.c_[f_mort_rates, f_mort_rates[:, 1] * m_transfer]

        return f_comps, f_transfer_rates, f_mort_rates

    INTs = {'flag_CHV': flag_CHV, 'flag_ANC': flag_ANC, 'flag_refer': flag_refer, 'flag_trans': flag_trans, 'flag_int1': flag_int1, 'flag_int2': flag_int2,
            'flag_int3': flag_int3, 'flag_int4': flag_int4, 'flag_int5': flag_int5, 'flag_int6': flag_int6, 'flag_sdr': flag_sdr, 'flag_capacity': flag_capacity,
            'flag_community': flag_community, 'flag_refer_capacity': flag_refer_capacity}
    b_INTs = {'flag_CHV': 0, 'flag_ANC': 0, 'flag_refer': 0, 'flag_trans': 0, 'flag_int1': 0, 'flag_int2': 0, 'flag_int3': 0, 'flag_int4': 0, 'flag_int5': 0,
              'flag_int6': 0, 'flag_sdr': 0, 'flag_capacity': 0, 'flag_community': 0, 'flag_refer_capacity': 0}
    n_months = 36
    t = range(n_months)
    time_comp = [0, stop_time]
    f_intvs = set_time_int(INTs, time_comp)
    b_f_intvs = set_time_int(b_INTs, time_comp)

    submitted = st.form_submit_button("Run Model")
    if submitted:
        b_global_vars = reset_global_vars()
        global_vars = {
            'ANCadded': ANCadded,
            'CHV_pushback': CHV_pushback,
            'CHV_cov': CHV_cov,
            'CHV_45': CHV_45,
            'know_added': know_added,
            'supply_added': {'supply_added_INT1': supply_added_INT1,
                             'supply_added_INT2': supply_added_INT2,
                             'supply_added_INT5': supply_added_INT5,
                             'supply_added_INT6': supply_added_INT6},
            'capacity_added': capacity_added,
            'transadded': transadded,
            'referadded': referadded,
            'ultracovhome': ultracovhome,
            'diagnosisrate': diagnosisrate,
            'refer_capacity': refer_capacity
        }
        b_outcomes = run_model(param, [0]*12, b_f_intvs, t, n_months, b_global_vars) #run_model([], b_flags)
        #b_outcomes
        outcomes = run_model(param, SDR_subcounties, f_intvs, t, n_months, global_vars) #run_model(SCID_selected, flags)
        b_df_aggregate = get_aggregate(b_outcomes)
        df_aggregate = get_aggregate(outcomes)
        df_years = agg_years(df_aggregate, n_months)
        b_df_years = agg_years(b_df_aggregate, n_months)
        df_outcomes = outcomes.dropna().reset_index()
        df_b_outcomes = b_outcomes.dropna().reset_index()

        ##########Genertate outcomes for plotting#########
        # # Subcounty level live birth and MMR and NMR
        LB_SC = np.concatenate(df_outcomes['Live Births Final'].values).reshape(-1, 4)
        LB_SC = np.column_stack((LB_SC, np.sum(LB_SC, axis=1)))
        LBtot = LB_SC[:, -1]
        bLB_SC = np.concatenate(df_b_outcomes['Live Births Final'].values).reshape(-1, 4)
        bLB_SC = np.column_stack((bLB_SC, np.sum(bLB_SC, axis=1)))
        bLBtot = bLB_SC[:, -1]

        bMM_SC = np.concatenate(df_b_outcomes['Deaths'].values).reshape(-1, 4)
        bMM_SC = np.array(np.column_stack((bMM_SC, np.sum(bMM_SC, axis=1))), dtype=np.float64)
        MM_SC = np.concatenate(df_outcomes['Deaths'].values).reshape(-1, 4)
        MM_SC = np.array(np.column_stack((MM_SC, np.sum(MM_SC, axis=1))), dtype=np.float64)
        bNM_SC = np.concatenate(df_b_outcomes['Neonatal Deaths'].values).reshape(-1, 4)
        bNM_SC = np.array(np.column_stack((bNM_SC, np.sum(bMM_SC, axis=1))), dtype=np.float64)
        NM_SC = np.concatenate(df_outcomes['Neonatal Deaths'].values).reshape(-1, 4)
        NM_SC = np.array(np.column_stack((NM_SC, np.sum(MM_SC, axis=1))), dtype=np.float64)

        MMR_SC = np.array(np.divide(MM_SC, LB_SC) * 1000, dtype=np.float64)
        bMMR_SC = np.array(np.divide(bMM_SC, bLB_SC) * 1000, dtype=np.float64)
        NMR_SC = np.array(np.divide(NM_SC, LB_SC) * 1000, dtype=np.float64)
        bNMR_SC = np.array(np.divide(bNM_SC, bLB_SC) * 1000, dtype=np.float64)

        bLB_SC = np.hstack((df_b_outcomes[['Subcounty', 'Time']], bLB_SC))
        LB_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], LB_SC))
        bMM_SC = np.hstack((df_b_outcomes[['Subcounty', 'Time']], bMM_SC))
        MM_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], MM_SC))
        bMMR_SC = np.hstack((df_b_outcomes[['Subcounty', 'Time']], bMMR_SC))
        MMR_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], MMR_SC))
        bNM_SC = np.hstack((df_b_outcomes[['Subcounty', 'Time']], bNM_SC))
        NM_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], NM_SC))
        bNMR_SC = np.hstack((df_b_outcomes[['Subcounty', 'Time']], bNMR_SC))
        NMR_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], NMR_SC))

        # #Subcounty level complications
        Com_SC = np.array([np.sum(arr, axis=1) for arr in df_outcomes['Complications-Health']], dtype=np.float64)
        ComR_SC = Com_SC / LBtot[:, np.newaxis] * 1000
        bCom_SC = np.array([np.sum(arr, axis=1) for arr in df_b_outcomes['Complications-Health']], dtype=np.float64)
        bComR_SC = bCom_SC / bLBtot[:, np.newaxis] * 1000

        Com_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], Com_SC))
        bCom_SC = np.hstack((df_b_outcomes[['Subcounty', 'Time']], bCom_SC))
        ComR_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], ComR_SC))
        bComR_SC = np.hstack((df_b_outcomes[['Subcounty', 'Time']], bComR_SC))
        LBtot = np.hstack((df_outcomes[['Subcounty', 'Time']], LBtot.reshape(-1, 1)))

        ##Process indicators
        bCapacityRatio_SC = df_b_outcomes[['Subcounty', 'Time', 'Capacity Ratio']]
        CapacityRatio_SC = df_outcomes[['Subcounty', 'Time', 'Capacity Ratio']]
        bANC = np.concatenate(df_b_outcomes['LB-ANC'].values).reshape(-1, 2)
        bANCrate = pd.DataFrame(bANC[:,-1] / np.sum(bANC, axis=1))
        bANCrate = np.hstack((df_b_outcomes[['Subcounty', 'Time']], bANCrate))
        ANC = np.concatenate(df_outcomes['LB-ANC'].values).reshape(-1, 2)
        ANCrate = pd.DataFrame(np.array(ANC[:, -1] / np.sum(ANC, axis=1)))
        ANCrate = np.hstack((df_outcomes[['Subcounty', 'Time']], ANCrate))
        bINT1Cov = df_b_outcomes[['Subcounty', 'Time', 'INT1 cover']]
        INT1Cov = df_outcomes[['Subcounty', 'Time', 'INT1 cover']]
        bINT2Cov = df_b_outcomes[['Subcounty', 'Time', 'INT2 cover']]
        INT2Cov = df_outcomes[['Subcounty', 'Time', 'INT2 cover']]
        bINT5Cov = df_b_outcomes[['Subcounty', 'Time', 'INT5 cover']]
        INT5Cov = df_outcomes[['Subcounty', 'Time', 'INT5 cover']]
        bINT6Cov = df_b_outcomes[['Subcounty', 'Time', 'INT6 cover']]
        INT6Cov = df_outcomes[['Subcounty', 'Time', 'INT6 cover']]
        normrefer = df_outcomes[['Subcounty', 'Time', 'Normal Referral']]
        bnormrefer = df_b_outcomes[['Subcounty', 'Time', 'Normal Referral']]
        comrefer = df_outcomes[['Subcounty', 'Time', 'Complication Referral']]
        bcomrefer = df_b_outcomes[['Subcounty', 'Time', 'Complication Referral']]
        capratiorefer = df_outcomes[['Subcounty', 'Time', 'Referral capacity ratio']]
        bcapratiorefer = df_b_outcomes[['Subcounty', 'Time', 'Referral capacity ratio']]

        ####Define functions for plotting
        faccols = ['Subcounty', 'month', 'Home', 'L2/3', 'L4', 'L5', 'Sum']
        comcols = ['Subcounty', 'month', 'PPH', 'Sepsis', 'Eclampsia', 'Obstructed', 'Others']
        def categorize_value(value):
            if value in options:
                return 'SDR'
            else:
                return 'Non-SDR'
        def lineplots (df, ptitle, y, ytitle, ymax):
            chart = (
                alt.Chart(
                    data=df,
                    title=ptitle,
                )
                .mark_line()
                .encode(
                    x=alt.X("month", axis=alt.Axis(title="Month")),
                    y=alt.Y(y, axis=alt.Axis(title=ytitle), scale=alt.Scale(domain=[0, ymax])
                            ),
                    color=alt.Color("level:N").title("Level"),
                ).properties(
                    width=400,
                    height=300
                )
            )

            chart = chart.properties(
            ).configure_title(
                anchor='middle'
            )
            return chart

        def pieplots (df0, ptitle):
            chart = (
                alt.Chart(
                    data=df0,
                    title=ptitle,
                )
                .mark_arc()
                .encode(
                    color='level:N',
                    theta='Percentage:Q',
                    tooltip=['level', 'Percentage']
                ).properties(
                    width=400,
                    height=300
                )
            )

            chart = chart.properties(
            ).configure_title(
                anchor='middle'
            )
            return chart

        def countylineplots(df0, cols, i, ytitle, ymax):

            df = pd.DataFrame(df0)
            df.columns = cols
            df = df.melt(id_vars=['Subcounty', 'month'], var_name='level', value_name='value')
            df_summary = df.groupby(['level', 'month'], as_index=False).sum()
            df_summary = df_summary[df_summary['level'].isin(selected_options)]

            chart = lineplots(df_summary, p_title[i], "value", ytitle, ymax)

            return st.altair_chart(chart)

        def barplots (df0, y, ytitle):

            chart = (
                alt.Chart(
                    data=df0,
                    title="Baseline vs Intervention in 3 years"
                )
                .mark_bar()
                .encode(
                    x=alt.X('Scenario:N', axis=None),
                    y=alt.Y(y + ':Q', axis=alt.Axis(title=ytitle)),
                    color=alt.Color('Scenario:N', scale=alt.Scale(domain=['Baseline', 'Intervention'],
                                                                  range=['#1f78b4', '#33a02c'])),
                    column=alt.Column('level:N', title=None, header=alt.Header(labelOrient='bottom')),
                    tooltip=['Scenario:N', y + ':Q']
                ).properties(
                    width=alt.Step(80),
                    height=300
                )
            )

            chart = chart.properties(
            ).configure_title(
                anchor='middle'
            )

            return chart

        def countypieplots(df0, cols, i):

            df = pd.DataFrame(df0)
            df.columns = cols
            df = df[df['month'] == 35]
            df = df.melt(id_vars=['Subcounty', 'month'], var_name='level', value_name='value')
            df_summary = df.groupby(['level'], as_index=False).sum()
            df_summary['Percentage'] = (df_summary['value'] / df_summary['value'].sum()) * 100

            chart = pieplots(df_summary, p_title[i])

            return st.altair_chart(chart)

        def countybarplots(basedf, intdf, cols, ytitle):
            df1 = pd.DataFrame(basedf)
            df2 = pd.DataFrame(intdf)
            df1.columns = cols
            df2.columns = cols
            #df1 = df1[df1['month'] == 35]
            #df2 = df2[df2['month'] == 35]
            df1['Scenario'] = 'Baseline'
            df2['Scenario'] = 'Intervention'
            df1 = df1.melt(id_vars=['Subcounty', 'month', 'Scenario'], var_name='level', value_name='value')
            df2 = df2.melt(id_vars=['Subcounty', 'month', 'Scenario'], var_name='level', value_name='value')
            df = pd.concat([df1, df2], ignore_index=True)
            df_summary = df.groupby(['level', 'Scenario'], as_index=False).sum()

            chart = barplots(df_summary, 'value', ytitle)

            return st.altair_chart(chart)

        def createcountyRatedf(LBdf, MMdf, cols):
            df1 = pd.DataFrame(LBdf)
            df2 = pd.DataFrame(MMdf)
            df1.columns = cols
            df1 = df1.melt(id_vars=['Subcounty', 'month'], var_name='level', value_name='value')
            df1 = df1.groupby(['level', 'month'])['value'].sum().reset_index()
            df1.rename(columns={df1.columns[2]: 'LB'}, inplace=True)
            df2.columns = cols
            df2 = df2.melt(id_vars=['Subcounty', 'month'], var_name='level', value_name='value')
            df2 = df2.groupby(['level', 'month'])['value'].sum().reset_index()
            df2.rename(columns={df2.columns[2]: 'MM'}, inplace=True)
            df = pd.merge(df1, df2, on=['level', 'month'], how='left')
            df['MMR'] = df['MM'] / df['LB'] * 1000
            Ratedf = df
            return Ratedf

        def countyRatelineplot(LBdf, MMdf, i, cols, ytitle, ymax):
            df = createcountyRatedf(LBdf, MMdf, cols)
            df = df[df['level'].isin(selected_options)]

            chart = lineplots(df, p_title[i], "MMR", ytitle, ymax)

            return st.altair_chart(chart)

        if selected_plotA == "Pathways":
            import plotly.graph_objects as go

            timepoint = 2
            b_lb_anc = b_df_years.loc[timepoint, 'LB-ANC']
            b_anc_anemia = b_df_years.loc[timepoint, 'ANC-Anemia']
            b_anemia_comp = b_df_years.loc[timepoint, 'Anemia-Complications'].T
            b_comp_health = b_df_years.loc[timepoint, 'Complications-Health']
            b_lb_anc = b_lb_anc.reshape((1, 2))
            b_lb_anc = pd.DataFrame(b_lb_anc, columns=['no ANC', 'ANC'], index=['Mothers'])
            b_anc_anemia = pd.DataFrame(b_anc_anemia, columns=['no Anemia', 'Anemia'], index=['no ANC', 'ANC'])
            b_anemia_comp = pd.DataFrame(b_anemia_comp,
                                         columns=['PPH', 'Sepsis', 'Eclampsia', 'Obstructed Labor', 'Other'],
                                         index=['no Anemia', 'Anemia'])
            b_comp_health = pd.DataFrame(b_comp_health, columns=['Unhealthy', 'Healthy'],
                                         index=['PPH', 'Sepsis', 'Eclampsia', 'Obstructed Labor', 'Other'])

            b_anc_anemia = b_anc_anemia.div(b_anc_anemia.sum(axis=0), axis=1) * np.array(b_anemia_comp.sum(axis=1))
            b_lb_anc.iloc[:, :] = np.array(b_anc_anemia.sum(axis=1))

            b_m_lb = np.round(b_df_years.loc[timepoint, 'Facility Level-Complications Pre'].astype(float),
                              decimals=0).reshape(1, 4)
            b_lb_lb = np.round(b_df_years.loc[timepoint, 'Complications-Complications'].astype(float), decimals=0)
            b_q_outcomes = np.round(b_df_years.loc[timepoint, 'Facility Level-Health'].astype(float), decimals=0)
            b_m_lb = pd.DataFrame(b_m_lb, columns=['Home (I)', 'L2/3 (I)', 'L4 (I)', 'L5 (I)'], index=['Mothers'])
            b_lb_lb = pd.DataFrame(b_lb_lb, columns=['Home (F)', 'L2/3 (F)', 'L4 (F)', 'L5 (F)'],
                                   index=['Home (I)', 'L2/3 (I)', 'L4 (I)', 'L5 (I)'])
            b_q_outcomes = pd.DataFrame(b_q_outcomes, columns=['Unhealthy', 'Healthy'],
                                        index=['Home (F)', 'L2/3 (F)', 'L4 (F)', 'L5 (F)'])

            lb_anc = df_years.loc[timepoint, 'LB-ANC']
            anc_anemia = df_years.loc[timepoint, 'ANC-Anemia']
            anemia_comp = df_years.loc[timepoint, 'Anemia-Complications'].T
            comp_health = df_years.loc[timepoint, 'Complications-Health']
            lb_anc = lb_anc.reshape((1, 2))
            lb_anc = pd.DataFrame(lb_anc, columns=['no ANC', 'ANC'], index=['Mothers'])
            anc_anemia = pd.DataFrame(anc_anemia, columns=['no Anemia', 'Anemia'], index=['no ANC', 'ANC'])
            anemia_comp = pd.DataFrame(anemia_comp, columns=['PPH', 'Sepsis', 'Eclampsia', 'Obstructed Labor', 'Other'],
                                       index=['no Anemia', 'Anemia'])
            comp_health = pd.DataFrame(comp_health, columns=['Unhealthy', 'Healthy'],
                                       index=['PPH', 'Sepsis', 'Eclampsia', 'Obstructed Labor', 'Other'])

            anc_anemia = anc_anemia.div(anc_anemia.sum(axis=0), axis=1) * np.array(anemia_comp.sum(axis=1))
            lb_anc.iloc[:, :] = np.array(anc_anemia.sum(axis=1))
            expand_out = comp_health[['Unhealthy']].T
            expand_out.columns = ['Deaths: PPH', 'Deaths: Sepsis', 'Deaths: Eclampsia', 'Deaths: Obstructed Labor',
                                  'Deaths: Other']

            nodes = ['Mothers', 'no ANC', 'ANC', 'no Anemia', 'Anemia', 'PPH', 'Sepsis', 'Eclampsia',
                     'Obstructed Labor', 'Other', 'Unhealthy', 'Healthy', 'Deaths: PPH', 'Deaths: Sepsis',
                     'Deaths: Eclampsia', 'Deaths: Obstructed Labor', 'Deaths: Other']
            # nodes = ['Mothers', 'no ANC', 'ANC', 'no Anemia', 'Anemia', 'PPH', 'Sepsis', 'Eclampsia', 'Obstructed Labor', 'Normal']
            sankey_dict = {x: i for i, x in enumerate(nodes)}

            source = []
            target = []
            value = []
            adj_matx = [b_lb_anc, b_anc_anemia, b_anemia_comp, b_comp_health, expand_out]
            # adj_matx = [lb_anc, anc_anemia, anemia_comp]
            for z, matx in enumerate(adj_matx):
                for i in range(matx.shape[0]):
                    for j in range(matx.shape[1]):
                        source.append(sankey_dict[matx.index[i]])
                        target.append(sankey_dict[matx.columns[j]])
                        value.append((matx.iloc[i, j]))

            fig = go.Figure(data=[go.Sankey(
                arrangement='snap',
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=nodes,
                    x=[0, 1 / 5, 1 / 5, 2 / 5, 2 / 5, 3 / 5, 3 / 5, 3 / 5, 3 / 5, 3 / 5, 4 / 5, 4 / 5, 5 / 5, 5 / 5,
                       5 / 5, 5 / 5],
                    y=[1, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )])

            # Update the layout
            fig.update_layout(title_text="Baseline",
                              font_size=10,
                              autosize=False,
                              width=600,
                              height=500)

            fig1_b = fig

            source = []
            target = []
            value = []
            adj_matx = [lb_anc, anc_anemia, anemia_comp, comp_health, expand_out]
            # adj_matx = [lb_anc, anc_anemia, anemia_comp]
            for z, matx in enumerate(adj_matx):
                for i in range(matx.shape[0]):
                    for j in range(matx.shape[1]):
                        source.append(sankey_dict[matx.index[i]])
                        target.append(sankey_dict[matx.columns[j]])
                        value.append(int(round(matx.iloc[i, j])))

            fig = go.Figure(data=[go.Sankey(
                arrangement='snap',
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=nodes,
                    x=[0, 1 / 5, 1 / 5, 2 / 5, 2 / 5, 3 / 5, 3 / 5, 3 / 5, 3 / 5, 3 / 5, 4 / 5, 4 / 5, 5 / 5, 5 / 5,
                       5 / 5, 5 / 5],
                    y=[1, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )])

            # Update the layout
            fig.update_layout(title_text="Intervention",
                              font_size=10,
                              autosize=False,
                              width=600,
                              height=500)

            # Show the plot
            fig1 = fig

            ########################################################################################################################

            m_lb = np.round(df_years.loc[timepoint, 'Facility Level-Complications Pre'].astype(float),
                            decimals=0).reshape(1, 4)
            lb_lb = np.round(df_years.loc[timepoint, 'Complications-Complications'].astype(float), decimals=0)
            q_outcomes = np.round(df_years.loc[timepoint, 'Facility Level-Health'].astype(float), decimals=0)
            m_lb = pd.DataFrame(m_lb, columns=['Home (I)', 'L2/3 (I)', 'L4 (I)', 'L5 (I)'], index=['Mothers'])
            lb_lb = pd.DataFrame(lb_lb, columns=['Home (F)', 'L2/3 (F)', 'L4 (F)', 'L5 (F)'],
                                 index=['Home (I)', 'L2/3 (I)', 'L4 (I)', 'L5 (I)'])
            q_outcomes = pd.DataFrame(q_outcomes, columns=['Unhealthy', 'Healthy'],
                                      index=['Home (F)', 'L2/3 (F)', 'L4 (F)', 'L5 (F)'])
            expand_out = q_outcomes[['Unhealthy']].T
            expand_out.columns = ['Deaths at Home', 'Deaths at L2/3', 'Deaths at L4', 'Deaths at L5']

            nodes = ['Mothers', 'Home (I)', 'L2/3 (I)', 'L4 (I)', 'L5 (I)', 'Home (F)', 'L2/3 (F)', 'L4 (F)', 'L5 (F)',
                     'Unhealthy', 'Healthy', 'Deaths at Home', 'Deaths at L2/3', 'Deaths at L4', 'Deaths at L5']
            sankey_dict = {x: i for i, x in enumerate(nodes)}

            source = []
            target = []
            value = []
            adj_matx = [b_m_lb, b_lb_lb, b_q_outcomes, expand_out]
            for z, matx in enumerate(adj_matx):
                for i in range(matx.shape[0]):
                    for j in range(matx.shape[1]):
                        source.append(sankey_dict[matx.index[i]])
                        target.append(sankey_dict[matx.columns[j]])
                        value.append(matx.iloc[i, j])

                        # Define the nodes
            # Create the Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                arrangement='snap',
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=nodes,
                    x=[0, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 3 / 4, 3 / 4, 1, 1, 1, 1],
                    y=[1, 0, 0.25 * 1, 0.25 * 2, 0.25 * 3, 0, 0.25 * 1, 0.25 * 2, 0.25 * 3, 0, 0.5, 0.8, 0.8, 0.8, 0.8]
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )])

            # Update the layout
            fig.update_layout(title_text="Baseline",
                              font_size=10,
                              autosize=False,
                              width=600,
                              height=500)

            fig2_b = fig

            source = []
            target = []
            value = []
            adj_matx = [m_lb, lb_lb, q_outcomes, expand_out]
            for z, matx in enumerate(adj_matx):
                for i in range(matx.shape[0]):
                    for j in range(matx.shape[1]):
                        source.append(sankey_dict[matx.index[i]])
                        target.append(sankey_dict[matx.columns[j]])
                        value.append(matx.iloc[i, j])

                        # Define the nodes
            # Create the Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                arrangement='snap',
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=nodes,
                    x=[0, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 3 / 4, 3 / 4, 1, 1, 1, 1],
                    y=[1, 0, 0.25 * 1, 0.25 * 2, 0.25 * 3, 0, 0.25 * 1, 0.25 * 2, 0.25 * 3, 0, 0.5, 0.8, 0.8, 0.8, 0.8]
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )])

            # Update the layout
            fig.update_layout(title_text="Intervention",
                              font_size=10,
                              autosize=False,
                              width=600,
                              height=500)

            # Show the plot
            fig2 = fig

            ########################################################################################################################
            tab1, tab2 = st.tabs(["Complication pathway", "Facility pathway"])
            with tab1:
                st.markdown("<h3 style='text-align: left;'>Subset of Mothers' Health through Pregnancy, Labor, and Delivery</h3>",
                            unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig1_b)

                with col2:
                    st.plotly_chart(fig1)
                    st.caption(
                        '*Note, relationships assumed based on literature values for factors not explicitly measured in the data, i.e. antenatal care and anemia.')
                    if np.array(b_lb_anc - lb_anc)[0][0] > 0:
                        st.markdown(
                            f'The intervention increased antenatal care rate by ~ **{round((np.array(b_lb_anc - lb_anc)[0][0])/np.sum(b_lb_anc, axis = 1)[0], ndigits = 2) * 100}%** in 3rd year.')
                    if np.sum(np.array(b_comp_health - comp_health)[:, 0]) > 0:
                        st.markdown(
                            f'The intervention reduced the number of maternal deaths by ~ **{round(np.sum(np.array(b_comp_health - comp_health)[:, 0]))}** in 3rd year.')

            with tab2:
                st.markdown(
                    "<h3 style='text-align: left;'>Mothers with Complications through Facilities</h3>",
                    unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig2_b)

                with col2:
                    st.plotly_chart(fig2)
                    st.caption(
                        '*Note, relationships assumed based on literature values for factors not explicitly measured in the data, i.e. antenatal care and anemia.')
                    if np.sum(np.array(b_m_lb - m_lb)[0][2:4]) > 0:
                        st.markdown(
                            f'The intervention increased the number of live births at L4/5 facilities by ~ **{round(np.sum(np.array(b_m_lb - m_lb)[0][2:4]))}.**')
                    if np.sum(np.array(b_lb_lb - lb_lb)[0:2, 2:4]) < 0:
                        st.markdown(
                            f'The intervention reduced the number of transfers to L4/5 facilities by ~ **{-round(np.sum(np.array(b_lb_lb - lb_lb)[0:2, 2:4]))}.**')
                    if np.sum(np.array(b_q_outcomes - q_outcomes)[:, 0]) > 0:
                        st.markdown(
                            f'The intervention reduced the number of maternal deaths by ~ **{round(np.sum(np.array(b_q_outcomes - q_outcomes)[:, 0]))}** in 3rd year.')

        if selected_plotA == "Cost effectiveness":
            st.markdown("<h3 style='text-align: left;'>Select the interventions to compare cost-effectiveness</h3>",
                        unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.text('Increase single intervention by 20%')
                ce_int1 = st.checkbox('Obstetric drape for pph')
                ce_int2 = st.checkbox('IV iron infusion for anemia')
                ce_int5 = st.checkbox('MgSO4 for elampsia')
                ce_int6 = st.checkbox('Antibiotics for maternal sepsis')
                ce_int1256 = st.checkbox('All single interventions')
            with col2:
                st.text('SDR interventions with mixed levels of supply and demand')
                ce_sdr1 = st.checkbox('SDR (high demand + high supply)')
                ce_sdr2 = st.checkbox('SDR (high demand + low supply)')
                ce_sdr3 = st.checkbox('SDR (low demand + high supply)')
                ce_sdr4 = st.checkbox('SDR (low demand + low supply)')
            st.markdown("---")
            INTs = reset_INTs()
            b_f_intvs = set_time_int(INTs, time_comp)
            global_vars = reset_global_vars()
            b_df = run_model(param, [0]*12, b_f_intvs, t, n_months, global_vars)
            #b_df.dropna().reset_index().to_csv("/Users/tingtingji/Library/CloudStorage/Dropbox/Phd PolyU/My Projects/Postdoc in JHU/SDR_project/Dashboard/Output/Baseline.csv"，index=False)
            b_df_aggregate = get_aggregate(b_df)
            b_df_years = agg_years(b_df_aggregate, n_months)
            #op_base = get_cost_effectiveness(INTs, time_comp[1], b_df, b_df, b_df_years, b_df_years, global_vars)

            if ce_int1:
                INTs = reset_INTs()
                INTs['flag_int1'] = 1
                f_intvs = set_time_int(INTs, time_comp)
                global_vars = reset_global_vars()
                global_vars['supply_added']['supply_added_INT1'] = 20
                df = run_model(param, SDR_subcounties, f_intvs, t, n_months, global_vars)
                #df.dropna().reset_index().to_csv("/Users/tingtingji/Library/CloudStorage/Dropbox/Phd PolyU/My Projects/Postdoc in JHU/SDR_project/Dashboard/Output/INT1.csv", index = False)
                df_aggregate = get_aggregate(df)
                df_years = agg_years(df_aggregate, n_months)
                op_int1 = get_cost_effectiveness(INTs, time_comp[1], df_years, b_df_years, global_vars)
            else:
                op_int1 = [0] * 18

            if ce_int2:
                INTs = reset_INTs()
                INTs['flag_int2'] = 1
                f_intvs = set_time_int(INTs, time_comp)
                global_vars = reset_global_vars()
                global_vars['supply_added']['supply_added_INT2'] = 20
                df = run_model(param, SDR_subcounties, f_intvs, t, n_months, global_vars)
                #df.dropna().reset_index().to_csv("/Users/tingtingji/Library/CloudStorage/Dropbox/Phd PolyU/My Projects/Postdoc in JHU/SDR_project/Dashboard/Output/INT2.csv", index = False)
                df_aggregate = get_aggregate(df)
                df_years = agg_years(df_aggregate, n_months)
                op_int2 = get_cost_effectiveness(INTs, time_comp[1], df_years, b_df_years, global_vars)
            else:
                op_int2 = [0] * 18

            if ce_int5:
                INTs = reset_INTs()
                INTs['flag_int5'] = 1
                f_intvs = set_time_int(INTs, time_comp)
                global_vars =  reset_global_vars()
                global_vars['supply_added']['supply_added_INT5'] = 20
                df = run_model(param, SDR_subcounties, f_intvs, t, n_months, global_vars)
                #df.dropna().reset_index().to_csv("/Users/tingtingji/Library/CloudStorage/Dropbox/Phd PolyU/My Projects/Postdoc in JHU/SDR_project/Dashboard/Output/INT5.csv", index = False)
                df_aggregate = get_aggregate(df)
                df_years = agg_years(df_aggregate, n_months)
                op_int5 = get_cost_effectiveness(INTs, time_comp[1], df_years, b_df_years, global_vars)
            else:
                op_int5 = [0] * 18

            if ce_int6:
                INTs = reset_INTs()
                INTs['flag_int6'] = 1
                f_intvs = set_time_int(INTs, time_comp)
                global_vars = reset_global_vars()
                global_vars['supply_added']['supply_added_INT6'] = 20
                df = run_model(param, SDR_subcounties, f_intvs, t, n_months, global_vars)
                #df.dropna().reset_index().to_csv("/Users/tingtingji/Library/CloudStorage/Dropbox/Phd PolyU/My Projects/Postdoc in JHU/SDR_project/Dashboard/Output/INT6.csv", index = False)
                df_aggregate = get_aggregate(df)
                df_years = agg_years(df_aggregate, n_months)
                op_int6 = get_cost_effectiveness(INTs, time_comp[1], df_years, b_df_years, global_vars)
            else:
                op_int6 = [0] * 18

            if ce_int1256:
                INTs = reset_INTs()
                INTs['flag_int1'] = 1
                INTs['flag_int2'] = 1
                INTs['flag_int5'] = 1
                INTs['flag_int6'] = 1
                f_intvs = set_time_int(INTs, time_comp)
                global_vars = reset_global_vars()
                global_vars['supply_added']['supply_added_INT1'] = 20
                global_vars['supply_added']['supply_added_INT2'] = 20
                global_vars['supply_added']['supply_added_INT5'] = 20
                global_vars['supply_added']['supply_added_INT6'] = 20
                df = run_model(param, SDR_subcounties, f_intvs, t, n_months, global_vars)
                #df.dropna().reset_index().to_csv("/Users/tingtingji/Library/CloudStorage/Dropbox/Phd PolyU/My Projects/Postdoc in JHU/SDR_project/Dashboard/Output/INT1256.csv", index = False)
                df_aggregate = get_aggregate(df)
                df_years = agg_years(df_aggregate, n_months)
                op_int1256 = get_cost_effectiveness(INTs, time_comp[1], df_years, b_df_years, global_vars)
            else:
                op_int1256 = [0] * 18

            if ce_sdr1:
                INTs = reset_INTs()
                INTs['flag_sdr'] = 1
                f_intvs = set_time_int(INTs, time_comp)
                global_vars = reset_global_vars()
                global_vars = {
                    'ANCadded': 0.5,
                    'CHV_pushback': True,
                    'CHV_cov': 1,
                    'CHV_45': 0.04,
                    'know_added': 0.6,
                    'supply_added': {'supply_added_INT1': 100,
                                     'supply_added_INT2': 100,
                                     'supply_added_INT5': 100,
                                     'supply_added_INT6': 100},
                    'capacity_added': 0.8,
                    'transadded': 0,
                    'referadded': 0,
                    'ultracovhome': 0,
                    'diagnosisrate': 0,
                    'refer_capacity': 300
                }
                df = run_model(param, SDR_subcounties, f_intvs, t, n_months, global_vars)
                #df.dropna().reset_index().to_csv("/Users/tingtingji/Library/CloudStorage/Dropbox/Phd PolyU/My Projects/Postdoc in JHU/SDR_project/Dashboard/Output/SDR1.csv", index = False)
                df_aggregate = get_aggregate(df)
                df_years = agg_years(df_aggregate, n_months)
                op_sdr1 = get_cost_effectiveness(INTs, time_comp[1], df_years, b_df_years, global_vars)
            else:
                op_sdr1 = [0] * 18

            if ce_sdr2:
                INTs = reset_INTs()
                INTs['flag_sdr'] = 1
                f_intvs = set_time_int(INTs, time_comp)
                global_vars = reset_global_vars()
                global_vars = {
                    'ANCadded': 0.5,
                    'CHV_pushback': True,
                    'CHV_cov': 1,
                    'CHV_45': 0.04,
                    'know_added': 0.2,
                    'supply_added': {'supply_added_INT1': 5,
                                     'supply_added_INT2': 5,
                                     'supply_added_INT5': 5,
                                     'supply_added_INT6': 5},
                    'capacity_added': 0.2,
                    'transadded': 0,
                    'referadded': 0,
                    'ultracovhome': 0,
                    'diagnosisrate': 0,
                    'refer_capacity': 100
                }
                df = run_model(param, SDR_subcounties, f_intvs, t, n_months, global_vars)
                #df.dropna().reset_index().to_csv("/Users/tingtingji/Library/CloudStorage/Dropbox/Phd PolyU/My Projects/Postdoc in JHU/SDR_project/Dashboard/Output/SDR2.csv", index = False)
                df_aggregate = get_aggregate(df)
                df_years = agg_years(df_aggregate, n_months)
                op_sdr2 = get_cost_effectiveness(INTs, time_comp[1], df_years, b_df_years, global_vars)
            else:
                op_sdr2 = [0] * 18

            if ce_sdr3:
                INTs = reset_INTs()
                INTs['flag_sdr'] = 1
                f_intvs = set_time_int(INTs, time_comp)
                global_vars = reset_global_vars()
                global_vars = {
                    'ANCadded': 0.5,
                    'CHV_pushback': True,
                    'CHV_cov': 0.2,
                    'CHV_45': 0.04,
                    'know_added': 0.6,
                    'supply_added': {'supply_added_INT1': 100,
                                     'supply_added_INT2': 100,
                                     'supply_added_INT5': 100,
                                     'supply_added_INT6': 100},
                    'capacity_added': 0.8,
                    'transadded': 0,
                    'referadded': 0,
                    'ultracovhome': 0,
                    'diagnosisrate': 0,
                    'refer_capacity': 300
                }
                df = run_model(param, SDR_subcounties, f_intvs, t, n_months, global_vars)
                #df.dropna().reset_index().to_csv("/Users/tingtingji/Library/CloudStorage/Dropbox/Phd PolyU/My Projects/Postdoc in JHU/SDR_project/Dashboard/Output/SDR3.csv", index = False)
                df_aggregate = get_aggregate(df)
                df_years = agg_years(df_aggregate, n_months)
                op_sdr3 = get_cost_effectiveness(INTs, time_comp[1], df_years, b_df_years, global_vars)
            else:
                op_sdr3 = [0] * 18

            if ce_sdr4:
                INTs = reset_INTs()
                INTs['flag_sdr'] = 1
                f_intvs = set_time_int(INTs, time_comp)
                global_vars = reset_global_vars()
                global_vars = {
                    'ANCadded': 0.5,
                    'CHV_pushback': True,
                    'CHV_cov': 0.2,
                    'CHV_45': 0.04,
                    'know_added': 0.2,
                    'supply_added': {'supply_added_INT1': 5,
                                     'supply_added_INT2': 5,
                                     'supply_added_INT5': 5,
                                     'supply_added_INT6': 5},
                    'capacity_added': 0.2,
                    'transadded': 0,
                    'referadded': 0,
                    'ultracovhome': 0,
                    'diagnosisrate': 0,
                    'refer_capacity': 100
                }
                df = run_model(param, SDR_subcounties, f_intvs, t, n_months, global_vars)
                #df.dropna().reset_index().to_csv("/Users/tingtingji/Library/CloudStorage/Dropbox/Phd PolyU/My Projects/Postdoc in JHU/SDR_project/Dashboard/Output/SDR4.csv", index = False)
                df_aggregate = get_aggregate(df)
                df_years = agg_years(df_aggregate, n_months)
                op_sdr4 = get_cost_effectiveness(INTs, time_comp[1], df_years, b_df_years, global_vars)
            else:
                op_sdr4 = [0] * 18

            df_ce = pd.DataFrame({
                #'Baseline': op_base,
                'Obstetric Drape': op_int1,
                'IV Iron Infusion': op_int2,
                'MgSO4': op_int5,
                'Antibiotics for Sepsis': op_int6,
                'All single interventions': op_int1256,
                'SDR (high demand + high supply)': op_sdr1,
                'SDR (high demand + low supply)': op_sdr2,
                'SDR (low demand + high supply)': op_sdr3,
                'SDR (low demand + low supply)': op_sdr4,
            }).T
            df_ce.columns = ['Cost (USD)', 'ANC Cost', 'Intervention Cost', 'Equipment Cost',
                             'Delivery Cost', 'C-section Cost', 'Labor Cost', 'Infrastructure Cost',
                             'CHV Cost', 'Access Cost', 'PPH', 'Eclampsia', 'Sepsis', 'Obstructed', 'Others', 'Added ANC (%)', 'Added facility deliveries (%)', 'DALY averted']
            df_ce['Cost per DALY averted'] = df_ce['Cost (USD)'] / df_ce['DALY averted']
            df_ce.iloc[:,:] = df_ce.iloc[:,:].round(1)
            df_ce = df_ce.dropna()
            tab1, tab2, tab3 = st.tabs(['Table', 'Pie chart', 'Bar chart'])
            with tab1:
                df_ce
            with tab2:
                st.markdown("<h3 style='text-align: left;'>Cost distributions under different scenarios</h3>",
                            unsafe_allow_html=True)
                df_cost = df_ce[['ANC Cost', 'Intervention Cost', 'Equipment Cost',
                                 'Delivery Cost', 'C-section Cost', 'Labor Cost', 'Infrastructure Cost',
                                 'CHV Cost', 'Access Cost']]

                scenarios = df_cost.index.tolist()
                num_rows, num_cols = 3, 3
                plot_columns = [st.columns(num_cols) for _ in range(num_rows)]
                for j in range(len(scenarios)):
                    df = df_cost.iloc[[j]].melt(var_name='type', value_name='value')
                    df['Percentage'] = (df['value'] / df['value'].sum()) * 100
                    chart = (
                        alt.Chart(
                            data=df,
                            title=scenarios[j],
                        )
                        .mark_arc()
                        .encode(
                            color='type:N',
                            theta='Percentage:Q',
                            tooltip=['type', 'Percentage']
                        ).properties(
                            width=400,
                            height=300
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
            with tab3:
                st.text('To be continued')

        if selected_plotA == "Live births":
            st.markdown("<h3 style='text-align: left;'>Live births</h3>",
                        unsafe_allow_html=True)
            st.markdown("(Note: Interventions and parameters that can change this outcome are shown as follows:)")
            st.markdown("****SDR Demand:** Employ CHVs -> CHV coverage, CHV effect on delivery at L4/5")
            st.markdown("****SDR Supply:** Upgrade L4/5 -> Improve facility capacity; Upgrade rescue -> Improve referral capacity")

            tab1, tab2, tab3 = st.tabs(["Line plots", "Pie charts", "Bar charts"])
            with tab1:
                options = ['Home', 'L2/3', 'L4', 'L5']
                selected_options = st.multiselect('Select levels', options)
                p_title = ['Baseline', 'Intervention']
                col0, col1 = st.columns(2)
                with col0:
                    countylineplots(bLB_SC[:, :6], faccols[:6], 0, "Number of live births", 40000 / 12)
                with col1:
                    countylineplots(LB_SC[:, :6], faccols[:6], 1, "Number of live births", 40000 / 12)

            with tab2:
                p_title = ['Baseline in month 36', 'Intervention in month 36']
                col0, col1 = st.columns(2)
                with col0:
                    countypieplots(bLB_SC[:, :6], faccols[:6], 0)
                with col1:
                    countypieplots(LB_SC[:, :6], faccols[:6],1)

            with tab3:
                countybarplots(bLB_SC[:, :6], LB_SC[:, :6], faccols[:6], "Number of live births")

        if selected_plotA == "Complications":
            st.markdown("<h3 style='text-align: left;'>Complications</h3>",
                        unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["Line plots", "Bar charts"])
            with tab1:
                options = ['PPH', 'Sepsis', 'Eclampsia', 'Obstructed', 'Others']
                selected_options = st.multiselect('Select levels:', options)
                p_title = ['Baseline', 'Intervention']
                col0, col1 = st.columns(2)
                with col0:
                    countylineplots(bCom_SC, comcols, 0, "Number of complications", 1200 / 12)
                with col1:
                    countylineplots(Com_SC, comcols, 1, "Number of complications", 1200 / 12)
            with tab2:
                countybarplots(bCom_SC, Com_SC, comcols, "Number of complications")

        if selected_plotA == "Maternal deaths":
            st.markdown("<h3 style='text-align: left;'>Maternal deaths</h3>",
                        unsafe_allow_html=True)
            tab1, tab3 = st.tabs(["Line plots", "Bar charts"])
            with tab1:
                options = ['Home', 'L2/3', 'L4', 'L5']
                selected_options = st.multiselect('Select levels', options)
                p_title = ['Baseline', 'Intervention']
                col0, col1 = st.columns(2)
                with col0:
                    countylineplots(bMM_SC[:, :6], faccols[:6], 0, "Number of maternal deaths", 50 / 12)
                with col1:
                    countylineplots(MM_SC[:, :6], faccols[:6],1, "Number of maternal deaths", 50 / 12)

            with tab3:
                countybarplots(bMM_SC[:, :6], MM_SC[:, :6], faccols[:6], "Number of maternal deaths")

        if selected_plotA == "Neonatal deaths":
            st.markdown("<h3 style='text-align: left;'>Neonatal deaths</h3>",
                        unsafe_allow_html=True)
            tab1, tab3 = st.tabs(["Line plots", "Bar charts"])
            with tab1:
                options = ['Home', 'L2/3', 'L4', 'L5']
                selected_options = st.multiselect('Select levels', options)
                p_title = ['Baseline', 'Intervention']
                col0, col1 = st.columns(2)
                with col0:
                    countylineplots(bNM_SC[:, :6], faccols[:6], 0, "Number of neonatal deaths", 120)
                with col1:
                    countylineplots(NM_SC[:, :6], faccols[:6],1, "Number of neonatal deaths", 120)

            with tab3:
                countybarplots(bNM_SC[:, :6], NM_SC[:, :6], faccols[:6], "Number of neonatal deaths")

        if selected_plotA == "Complication rate":
            st.markdown("<h3 style='text-align: left;'>Complications per 1000 live births</h3>",
                        unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["Line plots", "Bar charts"])
            p_title = ['Baseline', 'Intervention']

            with tab1:
                options = comcols[2:7]
                selected_options = st.multiselect('Select levels', options)

                def createComRdf(LBdf, Comdf, cols):
                    df1 = pd.DataFrame(LBdf)
                    df2 = pd.DataFrame(Comdf)
                    df1.columns = ['Subcounty', 'month', 'LB']
                    df1 = df1.groupby(['month'])['LB'].sum().reset_index()
                    df2.columns = cols
                    df2 = df2.melt(id_vars=['Subcounty', 'month'], var_name='level', value_name='value')
                    df2 = df2.groupby(['level', 'month'])['value'].sum().reset_index()
                    df2.rename(columns={df2.columns[2]: 'Com'}, inplace=True)
                    df = pd.merge(df1, df2, on=['month'], how='left')
                    df['ComR'] = df['Com'] / df['LB'] * 1000
                    return df

                col0, col1 = st.columns(2)
                df1 = createComRdf(LBtot, bCom_SC, comcols)
                ymax = df1['ComR'].max()
                df1 = df1[df1['level'].isin(selected_options)]
                df2 = createComRdf(LBtot, Com_SC, comcols)
                df2 = df2[df2['level'].isin(selected_options)]

                with col0:
                    chart = lineplots(df1, p_title[0], "ComR", "Complication rate", ymax)
                    st.altair_chart(chart)
                with col1:
                    chart = lineplots(df2, p_title[1], "ComR", "Complication rate", ymax)
                    st.altair_chart(chart)
        def createcountyyearRatedf(LBdf, MMdf, cols):
            df1 = pd.DataFrame(LBdf)
            df2 = pd.DataFrame(MMdf)
            df1.columns = cols
            df1 = df1.melt(id_vars=['Subcounty', 'month'], var_name='level', value_name='value')
            df1 = df1.groupby(['level'])['value'].sum().reset_index()
            df1.rename(columns={df1.columns[1]: 'LB'}, inplace=True)
            df2.columns = cols
            df2 = df2.melt(id_vars=['Subcounty', 'month'], var_name='level', value_name='value')
            df2 = df2.groupby(['level'])['value'].sum().reset_index()
            df2.rename(columns={df2.columns[1]: 'MM'}, inplace=True)
            df = pd.merge(df1, df2, on=['level'], how='left')
            df['MMR'] = df['MM'] / df['LB'] * 1000
            Ratedf = df
            return Ratedf

        if selected_plotA == "MMR":
            st.markdown("<h3 style='text-align: left;'>Maternal deaths per 1000 live births (MMR)</h3>",
                        unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["Line plots", "Bar charts"])
            p_title = ['Baseline', 'Intervention']

            with tab1:
                options = ['Home', 'L2/3', 'L4', 'L5']
                selected_options = st.multiselect('Select levels', options)
                col0, col1 = st.columns(2)
                with col0:
                    countyRatelineplot(bLB_SC[:, :6], bMM_SC[:, :6],0, faccols[:6], "MMR", 2)
                with col1:
                    countyRatelineplot(LB_SC[:, :6], MM_SC[:, :6], 1, faccols[:6], "MMR", 2)

            with tab2:

                df1 = createcountyyearRatedf(bLB_SC[:, :6], bMM_SC[:, :6], faccols[:6])
                df2 = createcountyyearRatedf(LB_SC[:, :6], MM_SC[:, :6], faccols[:6])
                df1['Scenario'] = 'Baseline'
                df2['Scenario'] = 'Intervention'
                df = pd.concat([df1, df2], ignore_index=True)
                chart = barplots(df, "MMR", "MMR")
                st.altair_chart(chart)

        if selected_plotA == "NMR":
            st.markdown("<h3 style='text-align: left;'>Neonatal deaths per 1000 live births (NMR)</h3>",
                        unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["Line plots", "Bar charts"])
            p_title = ['Baseline', 'Intervention']

            with tab1:
                options = ['Home', 'L2/3', 'L4', 'L5']
                selected_options = st.multiselect('Select levels', options)
                col0, col1 = st.columns(2)
                with col0:
                    countyRatelineplot(bLB_SC[:, :6], bNM_SC[:, :6],0, faccols[:6], "NMR", 60)
                with col1:
                    countyRatelineplot(LB_SC[:, :6], NM_SC[:, :6], 1, faccols[:6], "NMR", 60)

            with tab2:

                df1 = createcountyyearRatedf(bLB_SC[:, :6], bNM_SC[:, :6], faccols[:6])
                df2 = createcountyyearRatedf(LB_SC[:, :6], NM_SC[:, :6], faccols[:6])
                df1['Scenario'] = 'Baseline'
                df2['Scenario'] = 'Intervention'
                df = pd.concat([df1, df2], ignore_index=True)
                chart = barplots(df, "MMR", "NMR")
                st.altair_chart(chart)

        def subcountyplots(df, ptitle, ytitle, ymax):
            df = pd.DataFrame(df)
            df.columns = ['Subcounty', 'month', 'value']
            df['Subcounty'] = df['Subcounty'].replace(SC_ID)

            chart = (
                alt.Chart(
                    data=df,
                    title=ptitle,
                )
                .mark_line()
                .encode(
                    x=alt.X("month", axis=alt.Axis(title="Month")),
                    y=alt.Y('value', axis=alt.Axis(title=ytitle), scale=alt.Scale(domain=[0, ymax])
                            ),
                    color=alt.Color("Subcounty:N").title("Subcounty"),
                ).properties(
                    width=500,
                    height=400
                )
            )

            chart = chart.properties(
            ).configure_title(
                anchor='middle'
            )
            return chart


        if selected_plotA == "ANC rate":
            st.markdown("<h3 style='text-align: left;'>ANC rate</h3>",
                        unsafe_allow_html=True)

            p_title = ['Baseline', 'Intervention']
            col0, col1 = st.columns(2)
            cols = [col0, col1]
            dfs = [bANCrate, ANCrate]
            #ymax = max(np.max(dfs[0][:,2]), np.max(dfs[1][:,2]))

            for i in range(2):
                chart = subcountyplots(dfs[i], p_title[i], "ANC rate", 1)
                with cols[i]:
                    st.altair_chart(chart)

        if selected_plotA == "Capacity ratio":
            st.markdown("<h3 style='text-align: left;'>Capacity Ratio</h3>",
                        unsafe_allow_html=True)
            p_title = ['Baseline', 'Intervention']
            col0, col1 = st.columns(2)
            cols = [col0, col1]
            dfs = [bCapacityRatio_SC, CapacityRatio_SC]
            #ymax = max(np.max(dfs[0][:,2]), np.max(dfs[1][:,2]))

            for i in range(2):
                chart = subcountyplots(dfs[i], p_title[i], "Capacity Ratio", 1.2)
                with cols[i]:
                    st.altair_chart(chart)

        if selected_plotA == "Intervention coverage":
            st.markdown("<h3 style='text-align: left;'>Intervention coverage</h3>",
                        unsafe_allow_html=True)
            tab1, tab2, tab3, tab4 = st.tabs(["Obstetric drape", "IV iron infusion", "MgSO4 for eclampsia", "Antibiotics for maternal sepsis"])
            tabs = [tab1, tab2, tab3, tab4]
            p_title = ['Baseline', 'Intervention']
            dfs = [
                [bINT1Cov, INT1Cov],
                [bINT2Cov, INT2Cov],
                [bINT5Cov, INT5Cov],
                [bINT6Cov, INT6Cov]
            ]
            for i in range(4):
                with tabs[i]:
                    col0, col1 = st.columns(2)
                    cols = [col0, col1]
                    ymax = max(dfs[i][0].iloc[:, 2].max(), dfs[i][1].iloc[:, 2].max()) + 0.1
                    for j in range(2):
                        chart = subcountyplots(dfs[i][j], p_title[j], "Coverage", ymax)
                        with cols[j]:
                            st.altair_chart(chart)

        if selected_plotA == "Referral from home to L45":
            st.markdown("<h3 style='text-align: left;'>Referral from home to L4/5</h3>",
                        unsafe_allow_html=True)

            p_title = ['Baseline', 'Intervention']
            col0, col1 = st.columns(2)
            cols = [col0, col1]
            dfs = [bnormrefer, normrefer]
            ymax = max(dfs[0].iloc[:, 2].max(), dfs[1].iloc[:, 2].max()) + 10

            for i in range(2):
                chart = subcountyplots(dfs[i], p_title[i], "Referral", ymax)
                with cols[i]:
                    st.altair_chart(chart)

        if selected_plotA == "Emergency transfer":
            st.markdown("<h3 style='text-align: left;'>Emergency transfer from Home or L2/3 to L4/5</h3>",
                        unsafe_allow_html=True)

            p_title = ['Baseline', 'Intervention']
            col0, col1 = st.columns(2)
            cols = [col0, col1]
            dfs = [bcomrefer, comrefer]
            ymax = max(dfs[0].iloc[:, 2].max(), dfs[1].iloc[:, 2].max()) + 0.1

            for i in range(2):
                chart = subcountyplots(dfs[i], p_title[i], "Emergency transfer", ymax)
                with cols[i]:
                    st.altair_chart(chart)

        if selected_plotA == "Referral capacity ratio":
            st.markdown("<h3 style='text-align: left;'>Referral Capacity Ratio</h3>",
                        unsafe_allow_html=True)

            p_title = ['Baseline', 'Intervention']
            col0, col1 = st.columns(2)
            cols = [col0, col1]
            dfs = [bcapratiorefer, capratiorefer]
            ymax = max(dfs[0].iloc[:, 2].max(), dfs[1].iloc[:, 2].max()) + 0.1

            for i in range(2):
                chart = subcountyplots(dfs[i], p_title[i], "Referral Capacity Ratio", ymax)
                with cols[i]:
                    st.altair_chart(chart)

        def creatsubcountydf(df0, cols, colrange):
            df = pd.DataFrame(df0)
            df.columns = cols[:colrange]
            df['Subcounty'] = df['Subcounty'].replace(SC_ID)
            df['SDR'] = df['Subcounty'].apply(lambda x: categorize_value(x))
            df = df.melt(id_vars=['Subcounty', 'SDR', 'month'], var_name='level', value_name='value')
            return df

        def SDRsubcountylineplots(i, cols, colindex, ytitle, ymax):
            df = creatsubcountydf(dfs[i], cols, colindex)
            df = df[df['Subcounty'] == selected_options]
            df = df[df['level'].isin(selected_levels)]
            chart = lineplots(df, p_title[i], "value", ytitle, ymax)
            return chart

        def SDRsubcountypieplots(i, cols, colindex):
            df = creatsubcountydf(dfs[i], cols, colindex)
            df = df[df['Subcounty'] == selected_options]
            df = df[df['month'] == 35]
            df['Percentage'] = (df['value'] / df['value'].sum()) * 100
            chart = pieplots(df, p_title[i])
            return chart

        def NonSDRsubcountypieplots(cols, colindex):
            df = creatsubcountydf(dfs[0], cols, colindex)
            df = df[df['SDR'] == 'Non-SDR']
            df = df.groupby(['month', 'level'], as_index=False).agg({'value': 'mean'})     #.mean()
            df = df[df['month'] == 35]
            df['Percentage'] = (df['value'] / df['value'].sum()) * 100
            chart = pieplots(df, p_title[2])
            return chart

        def SDRsubcountybarplots(cols, colindex, ytitle):
            df1 = creatsubcountydf(dfs[0], cols, colindex)
            df1['Scenario'] = selected_options + ": Baseline"
            df2 = creatsubcountydf(dfs[1], cols, colindex)
            df2['Scenario'] = selected_options + ": Intervention"
            df = pd.concat([df1, df2], ignore_index=True)
            df = df[df['Subcounty'] == selected_options]
            df = df.groupby(['level', 'Scenario'], as_index=False).agg({'value': 'sum'})
            df = df[['level', 'value', 'Scenario']]
            df3 = creatsubcountydf(dfs[0], cols, colindex)
            df3 = df3[df3['SDR'] == 'Non-SDR']
            df3 = df3.groupby(['Subcounty', 'level'], as_index=False).agg({'value': 'sum'})
            df3 = df3.groupby(['level'], as_index=False).agg({'value': 'mean'})
            df3['Scenario'] = 'Non-SDR subcounties'
            df = pd.concat([df, df3], ignore_index=True)

            chart = (
                alt.Chart(
                    data=df,
                    title=selected_options + ' (Baseline + Intervention) vs Non-SDR subcounties in 3 years'
                )
                .mark_bar()
                .encode(
                    x=alt.X('Scenario:N', axis=None),
                    y=alt.Y('value:Q', axis=alt.Axis(title=ytitle)),
                    color=alt.Color('Scenario:N', scale=alt.Scale(domain=[selected_options + ": Baseline",
                                                                          selected_options + ": Intervention",
                                                                          'Non-SDR subcounties'],
                                                                  range=['#1f78b4', '#33a02c', 'orange'])),
                    column=alt.Column('level:N', title=None, header=alt.Header(labelOrient='bottom')),
                    tooltip=['Scenario:N', 'value:Q']
                ).properties(
                    width=alt.Step(50),
                    height=300
                )
            )

            chart = chart.properties(
            ).configure_title(
                anchor='middle'
            )

            return st.altair_chart(chart)


        def SDRsubcountyRatebarplots(cols, colindex, ytitle):
            df1 = creatsubcountydf(dfs[0], cols, colindex)
            df1['Scenario'] = selected_options + ": Baseline"
            df2 = creatsubcountydf(dfs[1], cols, colindex)
            df2['Scenario'] = selected_options + ": Intervention"
            df = pd.concat([df1, df2], ignore_index=True)
            df = df[df['Subcounty'] == selected_options]
            df = df.groupby(['level', 'Scenario'], as_index=False).agg({'value': 'mean'})
            df = df[['level', 'value', 'Scenario']]
            df3 = creatsubcountydf(dfs[0], cols, colindex)
            df3 = df3[df3['SDR'] == 'Non-SDR']
            df3 = df3.groupby(['Subcounty', 'level'], as_index=False).agg({'value': 'mean'})
            df3 = df3.groupby(['level'], as_index=False).agg({'value': 'mean'})
            df3['Scenario'] = 'Non-SDR subcounties'
            df = pd.concat([df, df3], ignore_index=True)

            chart = (
                alt.Chart(
                    data=df,
                    title=selected_options + ' (Baseline + Intervention) vs Non-SDR subcounties in 3 years'
                )
                .mark_bar()
                .encode(
                    x=alt.X('Scenario:N', axis=None),
                    y=alt.Y('value:Q', axis=alt.Axis(title=ytitle)),
                    color=alt.Color('Scenario:N', scale=alt.Scale(domain=[selected_options + ": Baseline",
                                                                          selected_options + ": Intervention",
                                                                          'Non-SDR subcounties'],
                                                                  range=['#1f78b4', '#33a02c', 'orange'])),
                    column=alt.Column('level:N', title=None, header=alt.Header(labelOrient='bottom')),
                    tooltip=['Scenario:N', 'value:Q']
                ).properties(
                    width=alt.Step(50),
                    height=300
                )
            )

            chart = chart.properties(
            ).configure_title(
                anchor='middle'
            )

            return st.altair_chart(chart)

        def NonSDRsubcountylineplots(i, cols, colindex, ytitle, ymax):
            df = creatsubcountydf(dfs[0], cols, colindex)
            df = df[df['SDR'] == 'Non-SDR']
            df = df[df['level'].isin(selected_levels)]
            df = df.groupby(['month', 'level'], as_index=False).agg({'value': 'mean'})  #.mean()
            chart = lineplots(df, p_title[i], "value", ytitle, ymax)
            return chart

        if selected_plotB == "Maternal deaths":
            st.markdown("<h3 style='text-align: left;'>Maternal deaths of SDR subcounties</h3>", unsafe_allow_html=True)
            options = select_SCs
            selected_options = st.selectbox('Select one SDR subcounty:', options)
            p_title = [selected_options + ': Baseline', selected_options + ': Intervention', 'Average of Non-SDR subcounties']

            tab1, tab2 = st.tabs(["Line plots", "Bar charts"])
            dfs = [bMM_SC[:,:6], MM_SC[:,:6]]
            ymax = np.max(bMM_SC[:,2:6])
            with tab1:
                level_options = ['Home', 'L2/3', 'L4', 'L5']
                selected_levels = st.multiselect('Select levels:', level_options)

                col0, col1, col3 = st.columns(3)
                col = [col0, col1, col3]
                for i in range(2):
                    chart = SDRsubcountylineplots(i, faccols, 6, "Number of maternal deaths", ymax)
                    with col[i]:
                        st.altair_chart(chart)

                with col[2]:
                    chart = NonSDRsubcountylineplots(2, faccols,6, "Number of maternal deaths", ymax)
                    st.altair_chart(chart)
            with tab2:
                SDRsubcountybarplots(faccols,6, "Maternal deaths")

        if selected_plotB == "MMR":
            st.markdown("<h3 style='text-align: left;'>MMR of SDR subcounties</h3>", unsafe_allow_html=True)

            options = select_SCs
            selected_options = st.selectbox('Select one SDR subcounty:', options)
            p_title = [selected_options + ': Baseline', selected_options + ': Intervention', 'Average of Non-SDR subcounties']

            tab1, tab2 = st.tabs(["Line plots", "Bar charts"])
            dfs = [bMMR_SC, MMR_SC]
            ymax = np.max(bMMR_SC[:,2:7])
            with tab1:
                level_options = ['Home', 'L2/3', 'L4', 'L5', 'Sum']
                selected_levels = st.multiselect('Select levels:', level_options)

                col0, col1, col2 = st.columns(3)
                col = [col0, col1, col2]
                for i in range(2):
                    chart = SDRsubcountylineplots(i, faccols,7, "MMR", ymax)
                    with col[i]:
                        st.altair_chart(chart)

                with col[2]:
                    chart = NonSDRsubcountylineplots(2, faccols,7, "MMR", ymax)
                    st.altair_chart(chart)
            with tab2:

                SDRsubcountyRatebarplots(faccols,7, "MMR")

        if selected_plotB == "Live births":
            st.markdown("<h3 style='text-align: left;'>%Live births of SDR subcounties</h3>", unsafe_allow_html=True)
            options = select_SCs
            selected_options = st.selectbox('Select one SDR subcounty:', options)
            p_title = [selected_options + ': Baseline', selected_options + ': Intervention', 'Average of Non-SDR subcounties']
            tab1, tab2, tab3 = st.tabs(["Line plots", "Pie charts", "Bar charts"])
            dfs = [bLB_SC[:,:6], LB_SC[:,:6]]
            ymax = np.max(LB_SC[:,2:6])

            with tab1:
                level_options = ['Home', 'L2/3', 'L4', 'L5']
                selected_levels = st.multiselect('Select levels:', level_options)
                col0, col1, col2 = st.columns(3)
                col = [col0, col1, col2]
                for i in range(2):
                    chart = SDRsubcountylineplots(i, faccols, 6, "Number of live births", ymax)
                    with col[i]:
                        st.altair_chart(chart)
                with col[2]:
                    chart = NonSDRsubcountylineplots(2, faccols,6, "Number of live births", ymax)
                    st.altair_chart(chart)

            with tab2:
                p_title = [selected_options + ': Baseline in month 36', selected_options + ': Intervention in month 36',
                           'Average of Non-SDR subcounties in month 36']
                col0, col1, col2 = st.columns(3)
                col = [col0, col1, col2]
                for i in range(2):
                    chart = SDRsubcountypieplots(i, faccols,6)
                    with col[i]:
                        st.altair_chart(chart)
                with col[2]:
                    chart = NonSDRsubcountypieplots(faccols, 6)
                    st.altair_chart(chart)

            with tab3:
                SDRsubcountybarplots(faccols,6, "Number of live births")

        if selected_plotB == "Complications":
            st.markdown("<h3 style='text-align: left;'>Number of complications of SDR subcounties</h3>", unsafe_allow_html=True)
            options = select_SCs
            selected_options = st.selectbox('Select one SDR subcounty:', options)
            p_title = [selected_options + ': Baseline', selected_options + ': Intervention',
                       'Average of Non-SDR subcounties']

            tab1, tab2 = st.tabs(["Line plots", "Bar charts"])
            dfs = [bCom_SC, Com_SC]
            ymax = np.max(bCom_SC[:,2:7])
            with tab1:
                level_options = ['PPH', 'Sepsis', 'Eclampsia', 'Obstructed', 'Others']
                selected_levels = st.multiselect('Select levels:', level_options)

                col0, col1, col3 = st.columns(3)
                col = [col0, col1, col3]
                for i in range(2):
                    chart = SDRsubcountylineplots(i, comcols, 7, "Number of complications", ymax)
                    with col[i]:
                        st.altair_chart(chart)

                with col[2]:
                    chart = NonSDRsubcountylineplots(2, comcols, 7, "Number of complications", ymax)
                    st.altair_chart(chart)
            with tab2:
                SDRsubcountybarplots(comcols,7, "Number of complications")

        if selected_plotB == "Complication rate":
            st.markdown("<h3 style='text-align: left;'>Complication rate (per 1000 live births) of SDR subcounties</h3>", unsafe_allow_html=True)
            options = select_SCs
            selected_options = st.selectbox('Select one SDR subcounty:', options)
            p_title = [selected_options + ': Baseline', selected_options + ': Intervention',
                       'Average of Non-SDR subcounties']

            tab1, tab2 = st.tabs(["Line plots", "Bar charts"])
            dfs = [bComR_SC, ComR_SC]
            ymax = np.max(bComR_SC[:,2:7])
            with tab1:
                level_options = ['PPH', 'Sepsis', 'Eclampsia', 'Obstructed', 'Others']
                selected_levels = st.multiselect('Select levels:', level_options)

                col0, col1, col3 = st.columns(3)
                col = [col0, col1, col3]
                for i in range(2):
                    chart = SDRsubcountylineplots(i, comcols, 7, "Complication rate", ymax)
                    with col[i]:
                        st.altair_chart(chart)

                with col[2]:
                    chart = NonSDRsubcountylineplots(2, comcols, 7, "Complication rate", ymax)
                    st.altair_chart(chart)
            with tab2:
                SDRsubcountyRatebarplots(comcols,7, "Complication rate")

        if select_level == "Subcounty in Map":
            data2 = gpd.read_file(shapefile_path2)
            data2 = data2.loc[data2['county'] == 'Kakamega',]
            data2['subcounty'] = data2['subcounty'].str.split().str[0]

            if selected_plotC == "MMR":
                outcome_options = ('Home', 'L2/3', 'L4', 'L5', 'Sum')
                selected = st.selectbox('Select level of interest:', outcome_options)
                p_title = ['Baseline', 'Intervention']

                st.markdown("<h3 style='text-align: left;'>MMR of SDR subcounties</h3>",
                            unsafe_allow_html=True)

                col0, col1 = st.columns(2)
                col = [col0, col1]

                df = pd.DataFrame(bMMR_SC)
                df.columns = ['Subcounty', 'Month', 'Home', 'L2/3', 'L4', 'L5', 'Sum']
                df = df[df['Month'] == month]
                scalemin = math.floor(df[selected].min())
                scalemax = math.ceil(df[selected].max())

                def categorize_value(value):
                    if value in select_SCs:
                        return 'SDR'
                    else:
                        return 'Non-SDR'

                for i in range(2):
                    if i == 0:
                        df = pd.DataFrame(bMMR_SC)
                    else:
                        df = pd.DataFrame(MMR_SC)

                    df.columns = ['Subcounty', 'Month', 'Home', 'L2/3', 'L4', 'L5', 'Sum']
                    df['Subcounty'] = df['Subcounty'].replace(SC_ID)
                    df['SDR'] = df['Subcounty'].apply(lambda x: categorize_value(x))
                    df = df[df['Month'] == month]

                    data = data2
                    data = data.merge(df, left_on='subcounty', right_on='Subcounty')
                    data = data[data['SDR'] == 'SDR']

                    chart = px.choropleth_mapbox(data,
                                           geojson=data.geometry,
                                           locations=data.index,
                                           mapbox_style="open-street-map",
                                           color=data[selected],
                                           center={'lon': 34.785511, 'lat': 0.430930},
                                           zoom=8,
                                           color_continuous_scale="ylgnbu",  # You can choose any color scale you prefer
                                           range_color=(scalemin, scalemax)  # Define the range of colors
                                               )

                    scatter_geo = go.Scattermapbox(
                        lon=data.geometry.centroid.x,
                        lat=data.geometry.centroid.y,
                        text=data['subcounty'],
                        textfont={"color": "white", "size": 30, "family": "Courier New"},
                        mode='text',
                        textposition='top center',
                    )
                    chart.add_trace(scatter_geo)

                    with col[i]:
                        st.write(p_title[i])
                        st.plotly_chart(chart, theme="streamlit", use_container_width=True)

                st.markdown("<h3 style='text-align: left;'>MMR of Non-SDR subcounties</h3>",
                            unsafe_allow_html=True)

                col0, col1 = st.columns(2)
                col = [col0, col1]

                for i in range(2):
                    if i == 0:
                        df = pd.DataFrame(bMMR_SC)
                    else:
                        df = pd.DataFrame(MMR_SC)

                    df.columns = ['Subcounty', 'Month', 'Home', 'L2/3', 'L4', 'L5', 'Sum']
                    df['Subcounty'] = df['Subcounty'].replace(SC_ID)
                    df['SDR'] = df['Subcounty'].apply(lambda x: categorize_value(x))
                    df = df[df['Month'] == month]

                    data = data2
                    data = data.merge(df, left_on='subcounty', right_on='Subcounty')
                    data = data[data['SDR'] == 'Non-SDR']

                    chart = px.choropleth_mapbox(data,
                                                 geojson=data.geometry,
                                                 locations=data.index,
                                                 mapbox_style="open-street-map",
                                                 color=data[selected],
                                                 center={'lon': 34.785511, 'lat': 0.430930},
                                                 zoom=8,
                                                 color_continuous_scale="ylgnbu",
                                                 range_color=(scalemin, scalemax)  # Define the range of colors
                                                 )

                    scatter_geo = go.Scattermapbox(
                        lon=data.geometry.centroid.x,
                        lat=data.geometry.centroid.y,
                        text=data['subcounty'],
                        textfont={"color": "white", "size": 30, "family": "Courier New"},
                        mode='text',
                        textposition='top center',
                    )
                    chart.add_trace(scatter_geo)

                    with col[i]:
                        st.write(p_title[i])
                        st.plotly_chart(chart, theme="streamlit", use_container_width=True)


            if selected_plotC == "% deliveries in L4/5":
                p_title = ['Baseline', 'Intervention']

                st.markdown("<h3 style='text-align: left;'>% deliveries in L4/5 of SDR subcounties</h3>",
                            unsafe_allow_html=True)

                col0, col1 = st.columns(2)
                col = [col0, col1]

                def categorize_value(value):
                    if value in select_SCs:
                        return 'SDR'
                    else:
                        return 'Non-SDR'

                for i in range(2):
                    if i == 0:
                        df = pd.DataFrame(bLB_SC)
                    else:
                        df = pd.DataFrame(LB_SC)

                    df.columns = ['Subcounty', 'Month', 'Home', 'L2/3', 'L4', 'L5', 'Sum']
                    df['Subcounty'] = df['Subcounty'].replace(SC_ID)
                    df['SDR'] = df['Subcounty'].apply(lambda x: categorize_value(x))
                    df = df[df['Month'] == month]

                    data = data2
                    data = data.merge(df, left_on='subcounty', right_on='Subcounty')
                    data = data[data['SDR'] == 'SDR']
                    data['outcome'] = (data.L4 + data.L5) / data.Sum

                    chart = px.choropleth_mapbox(data,
                                           geojson=data.geometry,
                                           locations=data.index,
                                           mapbox_style="open-street-map",
                                           color=data.outcome,
                                           center={'lon': 34.785511, 'lat': 0.430930},
                                           zoom=8,
                                           color_continuous_scale="ylgnbu",  # You can choose any color scale you prefer
                                           range_color=(0, 1)  # Define the range of colors
                                               )

                    scatter_geo = go.Scattermapbox(
                        lon=data.geometry.centroid.x,
                        lat=data.geometry.centroid.y,
                        text=data['subcounty'],
                        textfont={"color": "white", "size": 30, "family": "Courier New"},
                        mode='text',
                        textposition='top center',
                    )
                    chart.add_trace(scatter_geo)

                    with col[i]:
                        st.write(p_title[i])
                        st.plotly_chart(chart, theme="streamlit", use_container_width=True)

                st.markdown("<h3 style='text-align: left;'>% deliveries in L4/5 of Non-SDR subcounties</h3>",
                            unsafe_allow_html=True)

                col0, col1 = st.columns(2)
                col = [col0, col1]


                for i in range(2):
                    if i == 0:
                        df = pd.DataFrame(bLB_SC)
                    else:
                        df = pd.DataFrame(LB_SC)

                    df.columns = ['Subcounty', 'Month', 'Home', 'L2/3', 'L4', 'L5', 'Sum']
                    df['Subcounty'] = df['Subcounty'].replace(SC_ID)
                    df['SDR'] = df['Subcounty'].apply(lambda x: categorize_value(x))
                    df = df[df['Month'] == month]

                    data = data2
                    data = data.merge(df, left_on='subcounty', right_on='Subcounty')
                    data = data[data['SDR'] == 'Non-SDR']
                    data['outcome'] = (data.L4 + data.L5) / data.Sum

                    chart = px.choropleth_mapbox(data,
                                                 geojson=data.geometry,
                                                 locations=data.index,
                                                 mapbox_style="open-street-map",
                                                 color=data.outcome,
                                                 center={'lon': 34.785511, 'lat': 0.430930},
                                                 zoom=8,
                                                 color_continuous_scale="ylgnbu",
                                                 range_color=(0, 1)  # Define the range of colors
                                                 )

                    scatter_geo = go.Scattermapbox(
                        lon=data.geometry.centroid.x,
                        lat=data.geometry.centroid.y,
                        text=data['subcounty'],
                        textfont={"color": "white", "size": 30, "family": "Courier New"},
                        mode='text',
                        textposition='top center',
                    )
                    chart.add_trace(scatter_geo)

                    with col[i]:
                        st.write(p_title[i])
                        st.plotly_chart(chart, theme="streamlit", use_container_width=True)
