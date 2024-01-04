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
diagnosisrate = 0
ultracovhome = 0
CHV_cov = 0
CHV_45 = 0
CHV_ANC = 0
CHV_pushback = 0
referadded = 0
transadded = 0
capacity_added = 0
know_added = 0
supply_added = 0
ANCadded = 0


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

### Side bar ###
with st.sidebar:
    st.header("Outcomes sidebar")
    level_options = ("County", "Subcounty", "Subcounty in Map")
    select_level = st.selectbox('Select level of interest:', level_options)
    if select_level == "County":
        plotA_options = ("Pathways","Live births",
                         "Maternal deaths", "MMR",
                         "Cost effectiveness",
                         "Complications",
                         "Complication rate",
                         #"Quality of care"
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
                         #"Quality of care"
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

    SC_options = subcounties
    select_SCs = st.multiselect('Select subcounties to implement SDR policy:', SC_options)
    SCID_selected = [key for key, value in SC_ID.items() if value in select_SCs]
    SDR_subcounties = [1 if i in SCID_selected else 0 for i in range(12)]

st.subheader("SDR policy")
SDR_int = st.checkbox('SDR interventions (both supply and demand)')
if SDR_int:
    flag_sdr = 1
    CHV_pushback = 1
    CHV_cov = 1
    CHV_45 = 0.02
    know_added = 0.2
    supply_added = 0.2
    capacity_added = 0.2
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("SDR demand")
    demand_options = ("Test the impacts", "Apply interventions")
    select_demand = st.selectbox('Increase deliveries at L4/5:', demand_options)
    if select_demand == "Test the impacts":
        flag_ANC = 1
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            ANCadded = st.slider('Increased 4+ANCs rate', min_value=0.0, max_value=1.0, step=0.1, value=0.2)
        com_int = st.checkbox('Community effect')
        if com_int:
            flag_community = 1

    if select_demand == "Apply interventions":
        CHVint = st.checkbox('Employ CHVs at communities')
        if CHVint:
            flag_CHV = 1
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                CHV_cov = st.slider("CHV Coverage", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
                CHV_pushback = st.checkbox('Pushback effect')
            with col1_2:
                CHV_45 = st.slider("CHV effect on delivery at L4/5", min_value=0.00, max_value=0.10, step=0.01,
                                   value=0.02)

with col2:
    st.subheader("SDR supply")
    supply_options = ("Test the impacts", "Apply interventions")
    select_facility = st.selectbox('Improve health facility system:', supply_options)
    if select_facility == "Test the impacts":
        flag_sdr = 1
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            QOC_b = st.slider("Improve quality of care", min_value=0.0, max_value=1.0, step=0.1, value=0.2)
            know_added = QOC_b
            supply_added = QOC_b
        with col2_2:
            Cap_b = st.slider("Improve capacity", min_value=0.0, max_value=1.0, step=0.1, value=0.2)
            capacity_added = Cap_b

    if select_facility == "Apply interventions":
        flag_sdr = 1
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.text('Treatments')
            flag_int2 = st.checkbox('IV iron infusion for anemia')
            flag_int5 = st.checkbox('MgSO4 for eclampsia')
            flag_in6 = st.checkbox('Antibiotics for maternal sepsis')
            flag_int1 = st.checkbox('Obstetric drape for pph')
        with col2_2:
            know_added = st.slider("Improve knowledge of facility health workers", min_value=0.0, max_value=1.0, step=0.1, value=0.2)
            supply_added = st.slider("Improve supplies of treatments", min_value=0.0, max_value=1.0, step=0.1, value=0.2)
            capacity_added = st.slider("Improve capacities of facilities", min_value=0.0, max_value=1.0, step=0.1, value=0.2)
with col3:
    st.subheader("SDR referral")
    referral_options = ("Test the impacts", "Apply interventions")
    select_referral = st.selectbox('Test referral/rescue system:', referral_options)
    if select_referral == "Test the impacts":
        col3_1, col3_2 = st.columns(2)
        with col3_1:
            flag_trans = 1
            transadded = st.slider('% transfer increased', min_value=0.0, max_value=1.0, step=0.1, value=0.5)
        with col3_2:
            flag_refer = 1
            referadded = st.slider('% referral increased', min_value=0.0, max_value=1.0, step=0.1, value=0.5)
    if select_referral == "Apply interventions":
        int3 = st.checkbox('Employ portable ultrasounds')
        if int3:
            col3_1, col3_2 = st.columns(2)
            with col3_1:
                ultracovhome = st.slider('Coverage at communities', min_value=0.0, max_value=1.0, step=0.1, value=0.5)
            with col3_2:
                flag_int3 = 1
                diagnosisrate = st.slider('Coverage at L2/3', min_value=0.0, max_value=1.0, step=0.1, value=0.5)

global_vars = [ANCadded, CHV_pushback, CHV_cov, CHV_45, know_added, supply_added, capacity_added, transadded, referadded, ultracovhome, diagnosisrate]

with (st.form('Test')):
    ### PARAMETERs ###
    shapefile_path2 = 'ke_subcounty.shp'

    param = {
        'pre_comp_home': 0.0225359774,
        'pre_comp_l23': 0.0225359774,
        'pre_comp_l4': 0.0417,
        'pre_comp_l5': 0.07219193223,
        'complication_rate': 0.032,
        'p_comp_severe': 0.358,
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
        't_home_l4': 0.75,
        't_home_l5': 0.19,
        't_l23_l4': 14.14,
        't_l23_l5': 18.05,
        't_l4_l4': 0,
        't_l4_l5': 9.73,
        'm_l_home': 0.1,
        'm_l_l23': 0.1,
        'm_l_l4': 0.1,
        'm_l_l5': 0.1,
        'm_h_home': 25,
        'm_h_l23': 25,
        'm_h_l4': 6,
        'm_h_l5': 3,
        'p_pph_mort': 0.07058355,
        'p_sepsis_mort': 0.1855773323,
        'p_eclampsia_mort': 0.3285932084,
        'p_ol_mort': 0.02096039958,
        'p_other_mort': 0.3942855098,
        'i_int1': 0.6,
        'i_int2': 0.7,
        'i_int5': 0.59,
        'i_int6': 0.8,
        'i_int1_coverage': 0.64,
        'i_int2_coverage': 0.14,
        'i_int5_coverage': 0.8,
        'i_int6_coverage': 1,
    }

    n_months = 36
    t = np.arange(n_months)

    Capacity = np.array([3972, 3108, 1020, 3168, 2412, 6888, 2580, 2136, 936, 2772, 1524, 4668])

    ### MODEL PERTINENT ###
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
        param['i_int2_coverage'],
        0.37,
        param['i_int1_coverage'],
        param['i_int5_coverage'],
        param['i_int6_coverage'],
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

    sc = pd.DataFrame({
        'LB': sc_LB,
        'ANC': sc_ANC,
        'CLASS': sc_CLASS,
        'knowledge': sc_knowledge
    })

    i_scenarios = pd.DataFrame({
        'Supplies': i_supplies
    })

    timepoint = 1

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

    Capacity_ratio = np.zeros((n_months, sc_time['n']))
    Push_back = np.zeros((n_months, sc_time['n']))
    opinion = np.zeros((n_months, sc_time['n']))

    f_comps = np.array([param['pre_comp_home'], param['pre_comp_l23'], param['pre_comp_l4'], param['pre_comp_l5']])
    comps_overall = param['complication_rate']
    f_refer = comps_overall - f_comps

    f_transfer_comps = np.array([
        [-4, -39, 4, 40],
        [-7, -13, -2, 23],
        [-6, -29, -6, 40],
        [-6, -62, 4, 65]
    ])

    f_transfer_rates = np.array([
        [0.00, param['t_home_l23'], param['t_home_l4'], param['t_home_l5']],
        [0.00, 0.00, param['t_l23_l4'], param['t_l23_l5']],
        [0.00, 0.00, 0, param['t_l4_l5']],
        [0.00, 0.00, 0.00, 0.00]
    ]) / 100

    m_transfer = 2

    f_mort_rates = np.array([
        [param['m_l_home'], param['m_h_home']],
        [param['m_l_l23'], param['m_h_l23']],
        [param['m_l_l4'], param['m_h_l4']],
        [param['m_l_l5'], param['m_h_l5']]
    ]) / 100
    f_mort_rates = np.c_[f_mort_rates, f_mort_rates[:, 1] * m_transfer]

    # PPH, Sepsis, Eclampsia, Obstructed Labor
    comp_transfer_props = np.array([0.2590361446, 0.1204819277, 0.2108433735, 0.4096385542])

    b_comp_mort_rates = np.array(
        [param['p_pph_mort'], param['p_sepsis_mort'], param['p_eclampsia_mort'], param['p_ol_mort'],
         param['p_other_mort']])


    def reset_INT():
        INT = {
            'CHV': {
                'n_months_push_back': 2,
                'b_pushback': 5,
                'sdr_effect': np.zeros((12, n_months)),
            },
            'ANC': {
                'effect': np.ones((12, n_months))
            },
            'refer': {
                'effect': np.ones((12, n_months))
            },
            'transfer': {
                'effect': np.ones((12, n_months))
            },
            'int1': {
                'effect': np.ones((12, n_months)),
                'coverage': np.ones((12, n_months))
            },
            'int2': {
                'effect': 1,
                'coverage': 1
            },
            'int3': {
                'effect': np.ones((12, n_months)),
                'coverage': np.ones((12, n_months))
            },
            'int4': {
                'effect': np.ones((12, n_months)),
                'coverage': np.ones((12, n_months))
            },
            'int5': {
                'effect': np.ones((12, n_months)),
                'coverage': np.ones((12, n_months))
            },
            'int6': {
                'effect': np.zeros((12, n_months)),
                'coverage': np.ones((12, n_months))
            },
            'SDR': {
                'capacity': np.ones((12, n_months)),
                'quality': np.ones((12, n_months))
            },
            'Community': {
                'effect': np.zeros(n_months)
            }
        }

        return INT


    p_ol = param['p_comp_ol']
    p_other = param['p_comp_other']
    p_severe = param['p_comp_severe']


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


    def community_effect(x):
        if x - 0.5 < 0:
            x = -(1 - x)
        else:
            x = x
        return 0.000015 * (np.exp(10 * x) - np.exp(-10 * x))


    def set_flags(subcounty, flags, i_scenarios, sc, global_vars):

        Capacity = np.array([3972, 3198, 1020, 3168, 2412, 6798, 2580, 2136, 936, 2772, 1524, 4668])

        flag_CHV = flags[0]
        flag_ANC = flags[1]
        flag_ANC_time = flags[2]
        flag_refer = flags[3]
        flag_trans = flags[4]
        flag_int1 = flags[5]  # Obstetric drape
        flag_int2 = flags[6]  # Anemia reduction through IV Iron
        flag_int3 = flags[7]  # ultrasound
        flag_int5 = flags[8]  # MgSO4 for eclampsia
        flag_int6 = flags[9]  # antibiotics for maternal sepsis
        flag_sdr = flags[10]  # SDR
        flag_capacity = flags[11]
        flag_community = flags[12]

        INT = reset_INT()

        supplies = i_scenarios['Supplies']
        knowledge = sc['knowledge']

        supply_added = global_vars[5]
        know_added = global_vars[6]
        capacity_added = global_vars[6]
        CHV_45 = global_vars[3]
        CHV_cov = global_vars[2]
        ANCadded = global_vars[0]
        transadded = global_vars[7]
        referadded = global_vars[8]

        for i, x in zip(range(sc_time['n']), subcounty):
            knowl = np.array([k for k in knowledge])
            suppl = np.array([s for s in supplies])
            if x == 1:
                # effect = 0.2
                int_suppl = np.array([max((1 + supply_added) * s, 1) for s in supplies])  # (1 + effect)*suppl
                if flag_sdr:
                    # effect = 0.2
                    knowl = np.array([max((1 + know_added) * k, 1) for k in
                                      knowledge])  # np.array([((1+effect) * k) for k in knowledge])
                    suppl = np.array([max((1 + supply_added) * s, 1) for s in
                                      supplies])  # np.array([((1+effect) * s) for s in supplies])
                    flag_CHV = 1
                    flag_int1 = 1
                    flag_int2 = 1
                    flag_int5 = 1
                    flag_int6 = 1
                    flag_capacity = 1
                if flag_CHV:
                    INT['CHV']['sdr_effect'][i, :] = np.repeat(CHV_45 * CHV_cov, n_months)
                if flag_ANC:
                    INT['ANC']['effect'][i, :] = np.repeat(1 + ANCadded, n_months)
                if flag_ANC_time:
                    INT['ANC']['effect'][i, :] = 1 + np.linspace(0, ANCadded, n_months)
                if flag_refer:
                    INT['refer']['effect'][i, :] = np.repeat(1 + referadded, n_months)
                if flag_trans:
                    INT['transfer']['effect'][i, :] = 1 + transadded
                if flag_int1:
                    quality_factor = knowl[i] * int_suppl[2] - knowl[i] * suppl[2]
                    INT['int1']['effect'][i, :] = 1- ((1-np.repeat(param['i_int1'], n_months)) * quality_factor)
                    INT['int1']['coverage'][i, :] = quality_factor
                if flag_int2:
                    quality_factor = knowl[i] * int_suppl[0] - knowl[i] * suppl[0]
                    INT['int2']['effect'] = 1-((1-param['i_int2']) * quality_factor)
                    INT['int2']['coverage'] = quality_factor
                if flag_int5:
                    quality_factor = knowl[i] * int_suppl[3] - knowl[i] * suppl[3]
                    INT['int5']['effect'][i, :] = 1- ((1-np.repeat(param['i_int5'], n_months)) * quality_factor)
                    INT['int5']['coverage'][i, :] = quality_factor
                if flag_int6:
                    quality_factor = knowl[i] * int_suppl[4] - knowl[i] * suppl[4]
                    INT['int6']['effect'][i, :] = 1- ((1-np.repeat(param['i_int6'], n_months)) * quality_factor)
                    INT['int6']['coverage'][i, :] = quality_factor
                if flag_capacity:
                    Capacity[i] = Capacity[i] * (1 + capacity_added)
                if flag_community:
                    INT['Community']['effect'][i] = 1

            else:
                quality_factor = knowl[i] * suppl[2]
                INT['int1']['coverage'][i, :] = quality_factor
                quality_factor = knowl[i] * suppl[0]
                INT['int2']['coverage'] = quality_factor
                quality_factor = knowl[i] * suppl[3]
                INT['int5']['coverage'][i, :] = quality_factor
                quality_factor = knowl[i] * suppl[4]
                INT['int6']['coverage'][i, :] = quality_factor

                INT['CHV']['sdr_effect'][i, :] = np.repeat(0, n_months)
                INT['ANC']['effect'][i, :] = np.repeat(1, n_months)
                INT['refer']['effect'][i, :] = np.repeat(1,
                                                          n_months)  # transfer intervention cannot be subcounty specific
                # INT['transfer']['effect'] = np.ones((12, n_months))
                INT['int1']['effect'][i, :] = np.repeat(1, n_months)
                # INT['int2']['effect'] = 1                          # intervention 2 cannot be subcounty specific
                INT['int5']['effect'][i, :] = np.repeat(1, n_months)
                INT['int6']['effect'][i, :] = np.repeat(1, n_months)
                INT['Community']['effect'][i] = 0

        i_INT = INT
        return i_INT, Capacity

    def get_cost_effectiveness(flags, timepoint, df_aggregate, b_df_aggregate, global_vars):

        DALYs = [0.114, 0.324, 0.54]
        # cost = number of hemorrhage complications
        # DALYs = number of hemorrhage lives reduced from baseline
        flag_CHV = flags[0]
        flag_ANC = flags[1]
        flag_ANC_time = flags[2]
        flag_refer = flags[3]
        flag_trans = flags[4]
        flag_int1 = flags[5]  # Obstetric drape
        flag_int2 = flags[6]  # Anemia reduction through IV Iron
        flag_int3 = flags[7]  # ultrasound
        flag_int5 = flags[8]  # MgSO4 for eclampsia
        flag_int6 = flags[9]  # antibiotics for maternal sepsis
        flag_sdr = flags[10]  # SDR
        flag_capacity = flags[11]
        flag_community = flags[12]

        # effect = 0.2
        # i_INT = set_flags(subcounty, flags, i_scenarios, sc)
        supply_added = global_vars[5]

        low_pph = np.sum(b_df_aggregate.loc[timepoint, 'Complications-Health'], axis=1)[0] * \
                  np.array([1 - p_severe, p_severe])[0]
        high_pph = np.sum(b_df_aggregate.loc[timepoint, 'Complications-Health'], axis=1)[0] * \
                   np.array([1 - p_severe, p_severe])[1]
        maternal_deaths = np.sum(b_df_aggregate.loc[timepoint, 'Deaths'])

        b_DALYs = DALYs[0] * low_pph * 62.63 + DALYs[1] * high_pph * 62.63 + DALYs[2] * maternal_deaths * 62.63

        cost = 0

        low_pph = np.sum(df_aggregate.loc[timepoint, 'Complications-Health'], axis=1)[0] * \
                  np.array([1 - p_severe, p_severe])[0]
        high_pph = np.sum(df_aggregate.loc[timepoint, 'Complications-Health'], axis=1)[0] * \
                   np.array([1 - p_severe, p_severe])[1]
        maternal_deaths = np.sum(df_aggregate.loc[timepoint, 'Complications-Health'], axis=0)[0]
        i_DALYs = DALYs[0] * low_pph * 62.63 + DALYs[1] * high_pph * 62.63 + DALYs[2] * maternal_deaths * 62.63

        DALYs_averted = b_DALYs - i_DALYs
        if flag_sdr:
            cost += 2904051
        if flag_int1:
            cost += 1 * (low_pph + high_pph)
        if flag_int2:
            cost += 2.26 * 0.555 * np.sum(df_aggregate.loc[timepoint, 'Live Births Final']) * supply_added
        if flag_int5:
            eclampsia = np.sum(df_aggregate.loc[timepoint, 'Complications-Health'], axis=1)[2] * supply_added
            cost += 3 * eclampsia
        if flag_int6:
            sepsis = np.sum(df_aggregate.loc[timepoint, 'Complications-Health'], axis=1)[1] * supply_added
            cost += 12.30 * sepsis

        return cost, DALYs_averted


    def run_model(subcounty, flags, global_vars):

        i_INT, Capacity = set_flags(subcounty, flags, i_scenarios, sc, global_vars)

        p_anemia = param['p_comp_anemia']
        p_pph = param['p_comp_pph']
        p_sepsis = param['p_comp_sepsis']
        p_eclampsia = param['p_comp_eclampsia']

        i_subcounty = []
        i_time = []

        p_anc_anemia = []
        p_anemia_pph = []
        p_anemia_sepsis = []
        p_anemia_eclampsia = []

        anemia_pph = np.array(odds_prob(param['or_anemia_pph'], p_pph, p_anemia)) * i_INT['int2']['effect']
        anemia_sepsis = np.array(odds_prob(param['or_anemia_sepsis'], p_sepsis, p_anemia)) * i_INT['int2']['effect']
        anemia_eclampsia = np.array(odds_prob(param['or_anemia_eclampsia'], p_eclampsia, p_anemia)) * i_INT['int2'][
            'effect']

        p_anemia_pph = [anemia_pph] * (sc_time['n'] * n_months)
        p_anemia_sepsis = [anemia_sepsis] * (sc_time['n'] * n_months)
        p_anemia_eclampsia = [anemia_eclampsia] * (sc_time['n'] * n_months)
        i_time = [i for _ in range(sc_time['n']) for i in range(n_months)]
        i_subcounty = [j for j in range(sc_time['n']) for i in range(n_months)]

        if flags[2] != 1:
            p_anc_anemia = [odds_prob(param['or_anc_anemia'], p_anemia, 1 - sc['ANC'][j]) * (
                        1 - sc['ANC'][j] * i_INT['ANC']['effect'][j, 0]) / (1 - sc['ANC'][j])
                            for j in range(sc_time['n'])
                            for _ in range(n_months)]
        else:
            p_anc_anemia = [np.array(odds_prob(param['or_anc_anemia'], p_anemia, 1 - sc['ANC'][j])) * (
                        1 - sc['ANC'][j] * i_INT['ANC']['effect'][j, i]) / (1 - sc['ANC'][j]) for j in
                            range(sc_time['n']) for i in range(n_months)]

        probs = pd.DataFrame(
            {'Subcounty': i_subcounty, 'Time': i_time, 'p_anc_anemia': p_anc_anemia, 'p_anemia_pph': p_anemia_pph,
             'p_anemia_sepsis': p_anemia_sepsis, 'p_anemia_eclampsia': p_anemia_eclampsia})

        flag_pushback = global_vars[1]
        Push_back = np.zeros((n_months, sc_time['n']))

        index = pd.MultiIndex.from_product([range(sc_time['n']), range(n_months)], names=['Subcounty', 'Time'])

        df_3d = pd.DataFrame(index=index, columns=['Live Births Initial', 'Live Births Final', 'LB-ANC', 'ANC-Anemia',
                                                   'Anemia-Complications',
                                                   'Complications-Facility Level', 'Facility Level-Complications Pre',
                                                   'Transferred In', 'Transferred Out',
                                                   'Facility Level-Complications Post', 'Facility Level-Severity',
                                                   'Transfer Severity Mortality', 'Deaths', 'Facility Level-Health',
                                                   'Complications-Survive', 'Complications-Health',
                                                   'Complications-Complications',
                                                   'Capacity Ratio', 'Push Back'])
        for i in t:
            LB_tot_i = np.zeros(4)
            for j in range(sc_time['n']):

                if i > 0:
                    sc_time['LB1s'][j][i, :], Push_back[i, j] = f_LB_effect(sc_time['LB1s'][j][i - 1, :], i_INT,
                                                                            flag_pushback, j, i, Capacity_ratio[:, j],
                                                                            opinion[i, j])
                    LB_tot_i = np.maximum(sc_time['LB1s'][j][0, :], 1)
                    SDR_multiplier = sc_time['LB1s'][j][i, :] / LB_tot_i

                    anc = sc['ANC'][j] * i_INT['ANC']['effect'][j, i]
                    LB_tot, LB_tot_final, lb_anc, anc_anemia, anemia_comps_pre, comps_level_pre, f_comps_level_pre, transferred_in, transferred_out, f_comps_level_post, f_severe_post, f_state_mort, f_deaths, f_health, f_comp_survive, comps_health, comps_comps = f_MM(
                        LB_tot_i, anc, i_INT, SDR_multiplier, f_transfer_rates, probs, j, i)

                    df_3d.loc[(j, i), ['Live Births Initial', 'Live Births Final', 'LB-ANC', 'ANC-Anemia',
                                       'Anemia-Complications', 'Complications-Facility Level',
                                       'Facility Level-Complications Pre', 'Transferred In', 'Transferred Out',
                                       'Facility Level-Complications Post', 'Facility Level-Severity',
                                       'Transfer Severity Mortality', 'Deaths', 'Facility Level-Health',
                                       'Complications-Survive', 'Complications-Health', 'Complications-Complications',
                                       'Push Back']] = [LB_tot, LB_tot_final.astype(float),
                                                        np.round(lb_anc, decimals=2), anc_anemia, anemia_comps_pre,
                                                        comps_level_pre, f_comps_level_pre,
                                                        transferred_in.astype(float), transferred_out.astype(float),
                                                        f_comps_level_post, f_severe_post, f_state_mort, f_deaths,
                                                        f_health, f_comp_survive, comps_health, comps_comps,
                                                        Push_back[i, j]]

                Capacity_ratio[i, j] = np.sum(sc_time['LB1s'][j][i, 2:4]) / Capacity[j]
                opinion[i, j] = np.sum(sc_time['LB1s'][j][i, 2:4]) / np.sum(sc_time['LB1s'][j][i, :])
                df_3d.loc[(j, i), 'Capacity Ratio'] = Capacity_ratio[i, j]
                LB_tot_i += sc_time['LB1s'][j][i, :]

        return df_3d


    def f_LB_effect(SC_LB_previous, INT, flag_pushback, j, i, Capacity_ratio, opinion):
        INT_b = INT['CHV']['sdr_effect'][j, i]
        INT_c = INT['Community']['effect'][j] * community_effect(opinion) * (INT_b > 0)
        if flag_pushback:
            i_months = range(max(0, i - INT['CHV']['n_months_push_back']), i)
            mean_capacity_ratio = np.mean(Capacity_ratio[i_months])
            # push_back = max(0, mean_capacity_ratio-1)
            push_back = max(0, mean_capacity_ratio - 1)
            scale = np.exp(-push_back * INT['CHV']['b_pushback'])
            INT_b = INT_b * scale + INT_c
            effect = np.array([np.exp(-INT_b * np.array([1, 1])), 1 + INT_b * np.array([1, 1])]).reshape(1,
                                                                                                         4)  # effect of intervention on LB
            SC_LB = SC_LB_previous * effect  # new numbers of LB
            SC_LB = SC_LB / np.sum(SC_LB) * np.sum(SC_LB_previous)  # sum should equal the prior sum
            return SC_LB, push_back
        else:
            INT_b = INT_b + INT_c
            effect = np.array([np.exp(-INT_b * np.array([1, 1])), 1 + INT_b * np.array([1, 1])]).reshape(1,
                                                                                                         4)  # effect of intervention on LB
            SC_LB = SC_LB_previous * effect  # new numbers of LB
            SC_LB = SC_LB / np.sum(SC_LB) * np.sum(SC_LB_previous)  # sum should equal the prior sum
            return SC_LB, 0


    def f_MM(LB_tot, anc, INT, SDR_multiplier_0, f_transfer_rates, probs, j, i):

        LB_tot = np.array(LB_tot) * SDR_multiplier_0

        probs_row = probs.loc[(probs['Subcounty'] == j) & (probs['Time'] == i)].iloc[0]

        # Retrieve values using the variable
        p_anc_anemia = probs_row.at['p_anc_anemia']
        p_anemia_pph = probs_row.at['p_anemia_pph']
        p_anemia_sepsis = probs_row.at['p_anemia_sepsis']
        p_anemia_eclampsia = probs_row.at['p_anemia_eclampsia']

        original_comps = LB_tot * comps_overall
        new_comps = original_comps - LB_tot * f_refer * INT['refer']['effect'][j, i]
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
        comp_reduction[0, 2:4] = INT['int1']['effect'][j, i]  # PPH
        comp_reduction[2, 2:4] = INT['int5']['effect'][j, i]  # eclampsia
        comps_level_pre = np.sum(anemia_comps_pre, axis=1)[:,
                          np.newaxis] * f_comps_prop  # complications by facility level
        comps_level_pre = comps_level_pre * comp_reduction
        anemia_comp_props = anemia_comps_pre / (np.sum(anemia_comps_pre, axis=1)[:, np.newaxis])
        change = (np.sum(anemia_comps_pre, axis=1) - np.sum(comps_level_pre, axis=1))[:, np.newaxis] * anemia_comp_props
        anemia_comps_pre = anemia_comps_pre - change

        f_comps_level_pre = np.sum(comps_level_pre, axis=0)
        f_severe_pre = f_comps_level_pre[:, np.newaxis] * np.array([1 - p_severe, p_severe])
        transfer = (1 - sc['CLASS'][j]) + INT['transfer']['effect'][j, i] * sc['CLASS'][j]
        f_transfer_rates = f_transfer_rates * transfer
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
        mort_reduction[1] = INT['int6']['effect'][j, i]  # sepsis
        comps_health[:, 0] = comps_health[:, 0] - mort_reduction
        comps_health[:, 1] = comps_health[:, 1] + mort_reduction

        comps_comps = f_severe_pre[:, 1].reshape(4, 1) * f_transfer_rates
        for i in range(4):
            comps_comps[i, i] = f_comps_level_pre[i] - np.sum(comps_comps, axis=1)[i]

        LB_tot_final = LB_tot + transferred_in - transferred_out

        return LB_tot, LB_tot_final, lb_anc, anc_anemia, anemia_comps_pre, comps_level_pre, f_comps_level_pre, transferred_in, transferred_out, f_comps_level_post, f_severe_post, f_state_mort, f_deaths, f_health, f_comp_survive, comps_health, comps_comps

    flags = [flag_CHV, flag_ANC, flag_ANC_time, flag_refer, flag_trans, flag_int1, flag_int2, flag_int3, flag_int5, flag_int6, flag_sdr, flag_capacity, flag_community]
    b_flags = np.zeros(13)

    submitted = st.form_submit_button("Run Model")
    if submitted:
        b_outcomes = run_model([0] * 12, b_flags, global_vars) #run_model([], b_flags)
        outcomes = run_model(SDR_subcounties, flags, global_vars) #run_model(SCID_selected, flags)
        b_df_aggregate = get_aggregate(b_outcomes)
        df_aggregate = get_aggregate(outcomes)
        df_outcomes = outcomes.dropna().reset_index()
        df_b_outcomes = b_outcomes.dropna().reset_index()

        ##########Genertate outcomes for plotting#########
        # # Subcounty level live birth and MMR
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

        MMR_SC = np.array(np.divide(MM_SC, LB_SC) * 1000, dtype=np.float64)
        bMMR_SC = np.array(np.divide(bMM_SC, bLB_SC) * 1000, dtype=np.float64)

        bLB_SC = np.hstack((df_b_outcomes[['Subcounty', 'Time']], bLB_SC))
        LB_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], LB_SC))
        bMM_SC = np.hstack((df_b_outcomes[['Subcounty', 'Time']], bMM_SC))
        MM_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], MM_SC))
        bMMR_SC = np.hstack((df_b_outcomes[['Subcounty', 'Time']], bMMR_SC))
        MMR_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], MMR_SC))

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
                    title="Baseline vs Intervention"
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
            df1 = df1[df1['month'] == 35]
            df2 = df2[df2['month'] == 35]
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

            b_lb_anc = b_df_aggregate.loc[timepoint, 'LB-ANC']
            b_anc_anemia = b_df_aggregate.loc[timepoint, 'ANC-Anemia']
            b_anemia_comp = b_df_aggregate.loc[timepoint, 'Anemia-Complications'].T
            b_comp_health = b_df_aggregate.loc[timepoint, 'Complications-Health']
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

            b_m_lb = np.round(b_df_aggregate.loc[timepoint, 'Facility Level-Complications Pre'].astype(float),
                              decimals=0).reshape(1, 4)
            b_lb_lb = np.round(b_df_aggregate.loc[timepoint, 'Complications-Complications'].astype(float), decimals=0)
            b_q_outcomes = np.round(b_df_aggregate.loc[timepoint, 'Facility Level-Health'].astype(float), decimals=0)
            b_m_lb = pd.DataFrame(b_m_lb, columns=['Home (I)', 'L2/3 (I)', 'L4 (I)', 'L5 (I)'], index=['Mothers'])
            b_lb_lb = pd.DataFrame(b_lb_lb, columns=['Home (F)', 'L2/3 (F)', 'L4 (F)', 'L5 (F)'],
                                   index=['Home (I)', 'L2/3 (I)', 'L4 (I)', 'L5 (I)'])
            b_q_outcomes = pd.DataFrame(b_q_outcomes, columns=['Unhealthy', 'Healthy'],
                                        index=['Home (F)', 'L2/3 (F)', 'L4 (F)', 'L5 (F)'])

            lb_anc = df_aggregate.loc[timepoint, 'LB-ANC']
            anc_anemia = df_aggregate.loc[timepoint, 'ANC-Anemia']
            anemia_comp = df_aggregate.loc[timepoint, 'Anemia-Complications'].T
            comp_health = df_aggregate.loc[timepoint, 'Complications-Health']
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
                              width=800,
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
                              width=800,
                              height=500)

            # Show the plot
            fig1 = fig

            ########################################################################################################################

            m_lb = np.round(df_aggregate.loc[timepoint, 'Facility Level-Complications Pre'].astype(float),
                            decimals=0).reshape(1, 4)
            lb_lb = np.round(df_aggregate.loc[timepoint, 'Complications-Complications'].astype(float), decimals=0)
            q_outcomes = np.round(df_aggregate.loc[timepoint, 'Facility Level-Health'].astype(float), decimals=0)
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
                              width=800,
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
                              width=800,
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
                            f'The intervention increased antenatal care by ~ **{round(np.array(b_lb_anc - lb_anc)[0][0])}%**.')
                    if np.sum(np.array(b_comp_health - comp_health)[:, 0]) > 0:
                        st.markdown(
                            f'The intervention reduced the number of deaths by ~ **{round(np.sum(np.array(b_comp_health - comp_health)[:, 0]))}**.')

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
                            f'The intervention reduced the number of deaths by ~ **{round(np.sum(np.array(b_q_outcomes - q_outcomes)[:, 0]))}**.')


        if selected_plotA == "Live births":
            st.markdown("<h3 style='text-align: left;'>Live births</h3>",
                        unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["Line plots", "Pie charts", "Bar charts"])
            with tab1:
                options = ['Home', 'L2/3', 'L4', 'L5']
                selected_options = st.multiselect('Select levels', options)
                p_title = ['Baseline', 'Intervention']
                col0, col1 = st.columns(2)
                with col0:
                    countylineplots(bLB_SC[:, :6], faccols[:6], 0, "Number of live births", 40000)
                with col1:
                    countylineplots(LB_SC[:, :6], faccols[:6], 1, "Number of live births", 40000)

            with tab2:
                p_title = ['Baseline', 'Intervention']
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
                    countylineplots(bCom_SC, comcols, 0, "Number of complications", 1200)
                with col1:
                    countylineplots(Com_SC, comcols, 1, "Number of complications", 1200)
            with tab2:
                countybarplots(bCom_SC, Com_SC, comcols, "Number of complications")

        if selected_plotA == "Maternal deaths":

            st.markdown("<h3 style='text-align: left;'>Maternal deaths</h3>",
                        unsafe_allow_html=True)
            tab1, tab2, tab3 = st.tabs(["Line plots", "Pie charts", "Bar charts"])
            with tab1:
                options = ['Home', 'L2/3', 'L4', 'L5']
                selected_options = st.multiselect('Select levels', options)
                p_title = ['Baseline', 'Intervention']
                col0, col1 = st.columns(2)
                with col0:
                    countylineplots(bMM_SC[:, :6], faccols[:6], 0, "Number of maternal deaths", 50)
                with col1:
                    countylineplots(MM_SC[:, :6], faccols[:6],1, "Number of maternal deaths", 50)

            with tab2:
                p_title = ['Baseline', 'Intervention']
                col0, col1 = st.columns(2)
                with col0:
                    countypieplots(bMM_SC[:, :6], faccols[:6],0)
                with col1:
                    countypieplots(MM_SC[:, :6], faccols[:6],1)

            with tab3:
                countybarplots(bMM_SC[:, :6], MM_SC[:, :6], faccols[:6], "Number of maternal deaths")

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
                df1 = createcountyRatedf(bLB_SC[:, :6], bMM_SC[:, :6], faccols[:6])
                df2 = createcountyRatedf(LB_SC[:, :6], MM_SC[:, :6], faccols[:6])
                df1['Scenario'] = 'Baseline'
                df2['Scenario'] = 'Intervention'
                df = pd.concat([df1, df2], ignore_index=True)
                df = df[df['month'] == 35]

                chart = barplots(df, "MMR", "MMR")

                st.altair_chart(chart)

        if selected_plotA == "Cost effectiveness":
            base_flag = np.zeros(13)
            base_flag[sc_intervention['Obstetric Drape']] = 1
            intervention1 = base_flag

            base_flag = np.zeros(13)
            base_flag[sc_intervention['Antenatal Corticosteroids']] = 1
            intervention2 = base_flag

            base_flag = np.zeros(13)
            base_flag[sc_intervention['Magnesium Sulfate']] = 1
            intervention3 = base_flag

            base_flag = np.zeros(13)
            base_flag[sc_intervention['Antibiotics for Sepsis']] = 1
            intervention4 = base_flag

            base_flag = np.zeros(13)
            base_flag[sc_intervention['SDR']] = 1
            sdr_intervention = base_flag

            b_df = run_model([0]*12, np.zeros(13), global_vars)
            df = run_model(SDR_subcounties, intervention1, global_vars)
            df2 = run_model(SDR_subcounties, intervention2, global_vars)
            df3 = run_model(SDR_subcounties, intervention3, global_vars)
            df4 = run_model(SDR_subcounties, intervention4, global_vars)
            df_sdr = run_model(SDR_subcounties, sdr_intervention, global_vars)
            b_df_aggregate = get_aggregate(b_df)
            df1_aggregate = get_aggregate(df)
            df2_aggregate = get_aggregate(df2)
            df3_aggregate = get_aggregate(df3)
            df4_aggregate = get_aggregate(df4)
            df_sdr_aggregate = get_aggregate(df_sdr)

            ce_obstetric_drape = get_cost_effectiveness(intervention1, 35, df1_aggregate, b_df_aggregate, global_vars)
            ce_mgso4 = get_cost_effectiveness(intervention3, 35, df3_aggregate, b_df_aggregate, global_vars)
            ce_sepsis = get_cost_effectiveness(intervention4, 35, df4_aggregate, b_df_aggregate, global_vars)
            ce_sdr = get_cost_effectiveness(sdr_intervention, 35, df_sdr_aggregate, b_df_aggregate, global_vars)

            df_ce = pd.DataFrame({
                'Obstetric Drape': ce_obstetric_drape,
                'MgSO4': ce_mgso4,
                'Antibiotics for Sepsis': ce_sepsis,
                'SDR': ce_sdr
            }).T
            df_ce.columns = ['Cost (USD)', 'DALY averted']
            df_ce['Cost per DALY averted'] = df_ce['Cost (USD)'] / df_ce['DALY averted']
            df_ce = df_ce.applymap(lambda x: round(x, 2))
            df_ce

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
            df = df[df['month'] == 35]
            df = df[['month', 'level', 'value', 'Scenario']]
            df3 = df1
            df3 = df3[df3['SDR'] == 'Non-SDR']
            df3 = df3.groupby(['month', 'level'], as_index=False).agg({'value': 'mean'})
            df3 = df3[df3['month'] == 35]
            df3['Scenario'] = 'Non-SDR subcounties'
            df = pd.concat([df, df3], ignore_index=True)

            chart = (
                alt.Chart(
                    data=df,
                    title=selected_options + ' (Baseline + Intervention) vs Non-SDR subcounties'
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
                SDRsubcountybarplots(faccols,7, "MMR")

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
                SDRsubcountybarplots(comcols,7, "Complication rate")

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
