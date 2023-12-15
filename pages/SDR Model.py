import streamlit as st
from matplotlib import pyplot as plt
import geopandas as gpd
import numpy as np
from sympy import symbols, Eq, solve
import plotly.express as px
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import math

st.set_page_config(layout="wide")
options = {
    'Off': 0,
    'On': 1
}

selected_plotA = None
selected_plotB = None
selected_plotC = None
flag_sub = 0
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
                         "Cost effectiveness", "DALYs", "Cost")
        selected_plotA = st.selectbox(
            label="Select outcome of interest:",
            options=plotA_options,
        )
    if select_level == "Subcounty":
        flag_sub = 1
        plotB_options = ("MMR",
                         "Live births",
                         "Complication rate")
        selected_plotB = st.selectbox(
            label="Select outcome of interest:",
            options=plotB_options,
        )
    if select_level == "Subcounty in Map":
        flag_sub = 1
        plotC_options = ("MMR", "% deliveries in L4/5")
        selected_plotC = st.selectbox(
            label="Choose your interested outcomes in map:",
            options=plotC_options,
        )
        month = st.slider("Which month to show the outcome?",  min_value=1, max_value=35, step=1, value=35)

    SC_options = subcounties
    select_SCs = st.multiselect('Select subcounties to implement SDR policy:', SC_options)
    SCID_selected = [key for key, value in SC_ID.items() if value in select_SCs]

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
                #CHV_ANC = st.slider("CHV effect on having 4+ ANCs", min_value=0.00, max_value=0.10, step=0.01, value=0.02)

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
        #flag_ANC = 1
        flag_sdr = 1
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.text('Treatments')
            flag_int2 = st.checkbox('IV iron infusion for anemia')
            #int2_coverage = st.slider("IV iron infusion (reduce anemia): Coverage", min_value=0.000, max_value=1.000, step=0.100, value=0.075)
            flag_int5 = st.checkbox('MgSO4 for eclampsia')
            #int5_coverage = st.slider("MgSO4 for eclampsia: Coverage", min_value=0.000, max_value=1.000, step=0.100, value=0.200)
            flag_in6 = st.checkbox('Antibiotics for maternal sepsis')
            #int6_coverage = st.slider("Antibiotics for maternal sepsis: Coverage", min_value=0.000, max_value=1.000, step=0.100, value=0.543)
            #st.text('Treatments (Postpartum)')
            flag_int1 = st.checkbox('Obstetric drape for pph')
        with col2_2:
            #st.text('Knowledge of facility health workers')
            know_added = st.slider("Improve knowledge of facility health workers", min_value=0.0, max_value=1.0, step=0.1, value=0.2)
            #st.text('Supplies of treatments')
            supply_added = st.slider("Improve supplies of treatments", min_value=0.0, max_value=1.0, step=0.1, value=0.2)
            #int1_coverage = st.slider("Obstetric drape (reduce pph): Coverage", min_value=0.000, max_value=1.000,step=0.100, value=0.348)
            #st.text('Capacities of facilities')
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

with st.form('Test'):
    ### PARAMETERs ###
    shapefile_path2 = 'ke_subcounty.shp'

    comps = ['Low PPH', 'High PPH', 'Neonatal Deaths', 'Maternal Deaths']
    DALYs = [0.114, 0.324, 1, 0.54]
    DALY_dict = {x: 0 for x in comps}

    ### MODEL PERTINENT ###
    n_months = 36
    t = np.arange(n_months)

    Capacity = np.array([3972, 3108, 1020, 3168, 2412, 6888, 2580, 2136, 936, 2772, 1524, 4668])

    sc_LB = [
        [1906, 1675, 2245, 118],
        [1799, 674, 1579, 1],
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
        0.14,
        0.37,
        0.64,
        0.8,
        1
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
        'knowledge': sc_knowledge,
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

    Capacity_ratio = np.zeros((n_months, sc_time['n']))
    Push_back = np.zeros((n_months, sc_time['n']))

    f_comps = np.array([0.0225359774, 0.0225359774, 0.0417, 0.07219193223])
    comps_overall = 0.032
    f_refer = comps_overall - f_comps

    f_transfer_comps = np.array([
        [-4, -39, 4, 40],
        [-7, -13, -2, 23],
        [-6, -29, -6, 40],
        [-6, -62, 4, 65]
    ])

    f_transfer_rates = np.array([
        [0.00, 0.00, 0.75, 0.19],
        [0.00, 0.00, 14.14, 18.05],
        [0.00, 0.00, 0, 9.73],
        [0.00, 0.00, 0.00, 0.00]
    ]) / 100

    m_transfer = 2

    f_mort_rates = np.array([
        [0.10, 25.00],
        [0.10, 25.00],
        [0.10, 6.00],
        [0.10, 3.00]
    ]) / 100
    f_mort_rates = np.c_[f_mort_rates, f_mort_rates[:, 1] * m_transfer]

    # PPH, Sepsis, Eclampsia, Obstructed Labor
    comp_transfer_props = np.array([0.2590361446, 0.1204819277, 0.2108433735, 0.4096385542])

    b_comp_mort_rates = np.array([0.07058355, 0.1855773323, 0.3285932084, 0.02096039958, 0.3942855098])

    # equation = complication proportions * b_comp_mort_rates / comp_mort_factor to get proportion deaths for each

    # flag_int4 = flags[8]   # antenatal corticosteroids - reduction of neonatal mortality

    ### QUALITY
    # Types of quality care
    ## healthcare worker skills
    ### recognition of complications
    ### baseline recognition of complications = current state
    ### improved recognition of complications -> effect size
    ### implementation of the right interventions
    ### given capacity vs. treatment

    ### capacity effect on quality care
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
            'effect': np.ones((12, n_months))
        },
        'int2': {
            'effect': 1
        },
        'int3': {
            'effect': np.ones((12, n_months))
        },
        'int4': {
            'effect': np.ones((12, n_months))
        },
        'int5': {
            'effect': np.ones((12, n_months))
        },
        'int6': {
            'effect': np.zeros((12, n_months))
        },
        'SDR': {
            'capacity': np.ones((12, n_months)),
            'quality': np.ones((12, n_months))
        }
    }

    p_ol = 0.01
    p_other = 0.0044
    p_severe = 0.358

    ### global functions
    ### global functions
    def odds_prob(oddsratio, p_comp, p_expose):
        """get probability of complication given exposure given odds ratio"""
        oddsratio = oddsratio
        p_comp = p_comp
        p_expose = p_expose

        x, y = symbols('x y')
        eq1 = Eq(x / (1 - x) / (y / (1 - y)) - oddsratio, 0)
        eq2 = Eq(p_comp - p_expose * x - (1 - p_expose) * y, 0)
        solution = solve((eq1, eq2), (x, y))[0]
        return solution


    def get_aggregate(df):

        df_aggregate = pd.DataFrame(index=range(n_months), columns=df.columns)
        for i in range(n_months):
            for j in df.columns:
                df_aggregate.loc[i, j] = np.sum(np.array(df.loc[(slice(None), i), j]), axis=0)

        return df_aggregate


    def set_flags(subcounty, flags, i_supplies, sc):
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

        supplies = i_scenarios['Supplies']
        knowledge = sc['knowledge']

        for i in range(sc_time['n']):
            if i in subcounty:
                if flag_sdr:
                    #effect = 0.2
                    INT['CHV']['sdr_effect'][i, :] = np.repeat(CHV_45 * CHV_cov, n_months)
                    Capacity[i] = Capacity[i] * (1 + capacity_added)
                    knowl = [max((1 + know_added) * k, 1) for k in knowledge]
                    suppl = [max((1 + supply_added) * s, 1) for s in supplies]
                    # flag_int1 = 1
                    # flag_int2 = 1
                    # flag_int5 = 1
                    # flag_int6 = 1
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
                    quality_factor = knowl[i] * suppl[2]
                    INT['int1']['effect'][i, :] = np.repeat(0.6, n_months) * quality_factor
                if flag_int2:
                    quality_factor = knowl[i] * suppl[0]
                    INT['int2']['effect'] = 0.7 * quality_factor
                if flag_int5:
                    quality_factor = knowl[i] * suppl[3]
                    INT['int5']['effect'][i, :] = np.repeat(0.59, n_months) * quality_factor
                if flag_int6:
                    quality_factor = knowl[i] * suppl[4]
                    INT['int6']['effect'][i, :] = np.repeat(0.8, n_months) * quality_factor

        return INT

    def run_model(subcounty, flags, INT):

        INT = set_flags(subcounty, flags, i_supplies, sc)

        p_anemia = []
        for i in range(sc_time['n']):
            if i in subcounty:
                p_anemia.append(0.25 * INT['int2']['effect'])
            else:
                p_anemia.append(0.25)

        p_pph = 0.017
        p_sepsis = 0.00159
        p_eclampsia = 0.0046

        p_anc_anemia = []
        p_anemia_pph = []
        p_anemia_sepsis = []
        p_anemia_eclampsia = []
        for i in range(sc_time['n']):
            p_anc_anemia.append(odds_prob(2.26, p_anemia[i], 1 - sc['ANC'][i]))
            p_anemia_pph.append(odds_prob(3.54, p_pph, p_anemia[i]))
            p_anemia_sepsis.append(odds_prob(5.68, p_sepsis, p_anemia[i]))
            p_anemia_eclampsia.append(odds_prob(3.74, p_eclampsia, p_anemia[i]))

        probs = pd.DataFrame(
            {'p_anc_anemia': p_anc_anemia, 'p_anemia_pph': p_anemia_pph, 'p_anemia_sepsis': p_anemia_sepsis,
             'p_anemia_eclampsia': p_anemia_eclampsia})

        flag_pushback = CHV_pushback
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
                    sc_time['LB1s'][j][i, :], Push_back[i, j] = f_LB_effect(sc_time['LB1s'][j][i - 1, :], INT,
                                                                            flag_pushback, j, i, Capacity_ratio[:, j])
                    LB_tot_i = np.maximum(sc_time['LB1s'][j][0, :], 1)
                    SDR_multiplier = sc_time['LB1s'][j][i, :] / LB_tot_i

                    anc = sc['ANC'][j] * INT['ANC']['effect'][j, i]
                    LB_tot, LB_tot_final, lb_anc, anc_anemia, anemia_comps_pre, comps_level_pre, f_comps_level_pre, transferred_in, transferred_out, f_comps_level_post, f_severe_post, f_state_mort, f_deaths, f_health, f_comp_survive, comps_health, comps_comps = f_MM(
                        LB_tot_i, anc, INT, SDR_multiplier, f_transfer_rates, probs, j, i)

                    df_3d.loc[(j, i), 'Live Births Initial'] = LB_tot
                    df_3d.loc[(j, i), 'Live Births Final'] = LB_tot_final.astype(float)
                    df_3d.loc[(j, i), 'LB-ANC'] = np.round(lb_anc, decimals=2)
                    df_3d.loc[(j, i), 'ANC-Anemia'] = anc_anemia
                    df_3d.loc[(j, i), 'Anemia-Complications'] = anemia_comps_pre
                    df_3d.loc[(j, i), 'Complications-Facility Level'] = comps_level_pre
                    df_3d.loc[(j, i), 'Facility Level-Complications Pre'] = f_comps_level_pre
                    df_3d.loc[(j, i), 'Transferred In'] = transferred_in.astype(float)
                    df_3d.loc[(j, i), 'Transferred Out'] = transferred_out.astype(float)
                    df_3d.loc[(j, i), 'Facility Level-Complications Post'] = f_comps_level_post
                    df_3d.loc[(j, i), 'Facility Level-Severity'] = f_severe_post
                    df_3d.loc[(j, i), 'Transfer Severity Mortality'] = f_state_mort
                    df_3d.loc[(j, i), 'Deaths'] = f_deaths
                    df_3d.loc[(j, i), 'Facility Level-Health'] = f_health
                    df_3d.loc[(j, i), 'Complications-Survive'] = f_comp_survive
                    df_3d.loc[(j, i), 'Complications-Health'] = comps_health
                    df_3d.loc[(j, i), 'Complications-Complications'] = comps_comps
                    df_3d.loc[(j, i), 'Push Back'] = Push_back[i, j]

                Capacity_ratio[i, j] = np.sum(sc_time['LB1s'][j][i, 2:4]) / Capacity[j]
                df_3d.loc[(j, i), 'Capacity Ratio'] = Capacity_ratio[i, j]
                LB_tot_i += sc_time['LB1s'][j][i, :]

        return df_3d


    def f_LB_effect(SC_LB_previous, INT, flag_pushback, j, i, Capacity_ratio):
        INT_b = INT['CHV']['sdr_effect'][j, i]
        if flag_pushback:
            i_months = range(max(0, i - INT['CHV']['n_months_push_back']), i)
            mean_capacity_ratio = np.mean(Capacity_ratio[i_months])
            push_back = max(0, mean_capacity_ratio - 1)
            scale = np.exp(-push_back * INT['CHV']['b_pushback'])
            INT_b = INT_b * scale
            effect = np.array([np.exp(-INT_b * np.array([1, 1])), 1 + INT_b * np.array([1, 1])]).reshape(1,
                                                                                                         4)  # effect of intervention on LB
            SC_LB = SC_LB_previous * effect  # new numbers of LB
            SC_LB = SC_LB / np.sum(SC_LB) * np.sum(SC_LB_previous)  # sum should equal the prior sum
            return SC_LB, push_back
        else:
            effect = np.array([np.exp(-INT_b * np.array([1, 1])), 1 + INT_b * np.array([1, 1])]).reshape(1,
                                                                                                         4)  # effect of intervention on LB
            SC_LB = SC_LB_previous * effect  # new numbers of LB
            SC_LB = SC_LB / np.sum(SC_LB) * np.sum(SC_LB_previous)  # sum should equal the prior sum
            return SC_LB, 0


    def f_MM(LB_tot, anc, INT, SDR_multiplier_0, f_transfer_rates, probs, j, i):
        LB_tot = np.array(LB_tot) * SDR_multiplier_0

        p_anc_anemia = probs['p_anc_anemia'][j]
        p_anemia_pph = probs['p_anemia_pph'][j]
        p_anemia_sepsis = probs['p_anemia_sepsis'][j]
        p_anemia_eclampsia = probs['p_anemia_eclampsia'][j]

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

    flags = [flag_CHV, flag_ANC, flag_ANC_time, flag_refer, flag_trans,
             flag_int1, flag_int2, flag_int3, flag_int5, flag_int6, flag_sdr]
    b_flags = [0,0,0,0,0,
               0,0,0,0,0,0]
    b_subcounty = []
    subcounty = SCID_selected
    timepoint = 1

    submitted = st.form_submit_button("Run Model")
    if submitted:
        b_outcomes = run_model(b_subcounty, b_flags, INT)
        outcomes = run_model(subcounty, flags, INT)
        df_aggregate = get_aggregate(outcomes)
        df_outcomes = outcomes.dropna().reset_index()
        df_b_outcomes = b_outcomes.dropna().reset_index()

        if selected_plotA == "Pathways":
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
            fig.update_layout(title_text="Subset of Mothers' Health through Pregnancy, Labor, and Delivery",
                              font_size=10,
                              autosize=False,
                              width=800,
                              height=500)

            # Show the plot
            st.plotly_chart(fig)

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
            adj_matx = [m_lb, lb_lb, q_outcomes, expand_out]
            for z, matx in enumerate(adj_matx):
                for i in range(matx.shape[0]):
                    for j in range(matx.shape[1]):
                        source.append(sankey_dict[matx.index[i]])
                        target.append(sankey_dict[matx.columns[j]])
                        value.append(matx.iloc[i, j])

            import plotly.graph_objects as go

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
            fig.update_layout(title_text="Mothers with Complications through Facilities",
                              font_size=10,
                              autosize=False,
                              width=800,
                              height=500)

            # Show the plot
            st.plotly_chart(fig)

        ##########Genertate outcomes for plotting#########
        #Live births
        overall_lbs = np.zeros((n_months,4))
        for i in range(12):
            for j in range(35):
                for k in range(4):
                    overall_lbs[j+1,k] += b_outcomes['Live Births Final'][i][j+1][k]
        overall_lbs = overall_lbs[1:35,:]

        overall_lbs_i = np.zeros((n_months,4))
        for i in range(12):
            for j in range(35):
                for k in range(4):
                    overall_lbs_i[j+1,k] += outcomes['Live Births Final'][i][j+1][k]
        overall_lbs_i = overall_lbs_i[1:35,:]
        # #Deaths
        save3 = np.zeros((n_months,4))
        for i in range(12):
            for j in range(35):
                for k in range(4):
                    save3[j+1,k] += b_outcomes['Deaths'][i][j+1][k]

        save3 = save3[1:35,:]

        save4 = np.zeros((n_months,4))
        for i in range(12):
            for j in range(35):
                for k in range(4):
                    save4[j+1,k] += outcomes['Deaths'][i][j+1][k]

        save4 = save4[1:35,:]

        # #Rates
        MMR_b = np.divide(save3, overall_lbs)
        MMR_b *= 1000
        MMR_i = np.divide(save4, overall_lbs_i)
        MMR_i *= 1000

        # # DALYs
        # boutcomes_DALY = {x: 0 for x in comps}
        # boutcomes_DALY['Maternal Deaths'] = pd.to_numeric(np.sum(DALYs[3] * 62.63 * bn_MM[:, :, 0], axis=1)).round(0)
        # boutcomes_DALY['Neonatal Deaths'] = pd.to_numeric(np.sum(DALYs[2] * 60.25 * bn_nM[:, :, 0], axis=1)).round(0)
        # boutcomes_DALY['High PPH'] = pd.to_numeric(np.sum(DALYs[1] * 62.63 * overall_lbs, axis=1) * np.array(bL_pph_H, dtype=np.float64)).round(0)
        # boutcomes_DALY['Low PPH'] = pd.to_numeric(np.sum(DALYs[0] * 62.63 * overall_lbs, axis=1) * np.array(bL_pph_L, dtype=np.float64)).round(0)
        #
        # outcomes_DALY = {x: 0 for x in comps}
        # outcomes_DALY['Maternal Deaths'] = pd.to_numeric(np.sum(DALYs[3] * 62.63 * n_MM[:, :, 0], axis=1)).round(0)
        # outcomes_DALY['Neonatal Deaths'] = pd.to_numeric(np.sum(DALYs[2] * 60.25 * n_nM[:, :, 0], axis=1)).round(0)
        # if flag_int1:
        #     outcomes_DALY['High PPH'] = pd.to_numeric(np.sum(DALYs[1] * 62.63 * overall_lbs_i * np.array(iL_pph_H, dtype=np.float64), axis=1)).round(0)
        #     outcomes_DALY['Low PPH'] = pd.to_numeric(np.sum(DALYs[0] * 62.63 * overall_lbs_i * np.array(iL_pph_L, dtype=np.float64), axis=1)).round(0)
        # else:
        #     outcomes_DALY['High PPH'] = pd.to_numeric(np.sum(DALYs[1] * 62.63 * overall_lbs_i, axis=1) * np.array(iL_pph_H, dtype=np.float64)).round(0)
        #     outcomes_DALY['Low PPH'] = pd.to_numeric(np.sum(DALYs[0] * 62.63 * overall_lbs_i, axis=1) * np.array(iL_pph_L, dtype=np.float64)).round(0)
        #

        # # Subcounty level live birth and MMR
        LB_SC = np.concatenate(df_outcomes['Live Births Final'].values).reshape(-1, 4)
        LB_SC = np.column_stack((LB_SC, np.sum(LB_SC, axis=1)))
        MM_SC = np.concatenate(df_outcomes['Deaths'].values).reshape(-1, 4)
        MM_SC = np.column_stack((MM_SC, np.sum(MM_SC, axis=1)))
        bLB_SC = np.concatenate(df_b_outcomes['Live Births Final'].values).reshape(-1, 4)
        bLB_SC = np.column_stack((bLB_SC, np.sum(bLB_SC, axis=1)))
        bMM_SC = np.concatenate(df_b_outcomes['Deaths'].values).reshape(-1, 4)
        bMM_SC = np.column_stack((bMM_SC, np.sum(bMM_SC, axis=1)))

        MMR_SC = np.array(np.divide(MM_SC, LB_SC) * 1000, dtype=np.float64)
        MMR_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], MMR_SC))
        bMMR_SC = np.array(np.divide(bMM_SC, bLB_SC) * 1000, dtype=np.float64)
        bMMR_SC = np.hstack((df_b_outcomes[['Subcounty', 'Time']], bMMR_SC))

        bLB_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], bLB_SC))
        LB_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], LB_SC))

        # #Subcounty level complications
        Com_SC = np.array([np.sum(arr, axis=1) for arr in df_outcomes['Complications-Health']], dtype=np.float64)
        LBtot = LB_SC[:, -1]
        ComR_SC = Com_SC / LBtot[:, np.newaxis] * 1000
        ComR_SC = np.hstack((df_outcomes[['Subcounty', 'Time']], ComR_SC))

        bCom_SC = np.array([np.sum(arr, axis=1) for arr in df_b_outcomes['Complications-Health']], dtype=np.float64)
        bLBtot = bLB_SC[:, -1]
        bComR_SC = bCom_SC / bLBtot[:, np.newaxis] * 1000
        bComR_SC = np.hstack((df_b_outcomes[['Subcounty', 'Time']], bComR_SC))

        if selected_plotA == "Live births":
            st.markdown("<h3 style='text-align: left;'>Live births</h3>",
                        unsafe_allow_html=True)

            options = ['Home', 'L2/3', 'L4', 'L5']
            selected_options = st.multiselect('Select levels', options)
            p_title = ['Baseline', 'Intervention']
            col0, col1 = st.columns(2)
            col = [col0, col1]

            for i in range(2):
                if i == 0:
                    df = pd.DataFrame(overall_lbs)
                else:
                    df = pd.DataFrame(overall_lbs_i)

                df['month'] = range(1, 35)
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
                        data= df,
                        title= p_title[i],
                    )
                    .mark_line()
                    .encode(
                        x=alt.X("month", axis=alt.Axis(title="Month")),
                        y=alt.Y("value", axis=alt.Axis(title="# of Live births"), scale=alt.Scale(domain=[0, 30000])),
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

                with col[i]:
                    st.altair_chart(chart)

        if selected_plotA == "Maternal deaths":
            st.markdown("<h3 style='text-align: left;'>Maternal deaths</h3>",
                        unsafe_allow_html=True)

            options = ['Home', 'L2/3', 'L4', 'L5']
            selected_options = st.multiselect('Select levels', options)
            p_title = ['Baseline', 'Intervention']
            col0, col1 = st.columns(2)
            col = [col0, col1]

            for i in range(2):
                if i == 0:
                    df = pd.DataFrame(save3)
                else:
                    df = pd.DataFrame(save4)

                df['month'] = range(1, 35)
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
                        title=p_title[i],
                    )
                    .mark_line()
                    .encode(
                        x=alt.X("month", axis=alt.Axis(title="Month")),
                        y=alt.Y("value", axis=alt.Axis(title="# of Maternal deaths"), scale=alt.Scale(domain=[0, 50])),
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

                with col[i]:
                    st.altair_chart(chart)

        if selected_plotA == "MMR":
            st.markdown("<h3 style='text-align: left;'>Maternal deaths per 1000 live births (MMR)</h3>",
                        unsafe_allow_html=True)

            options = ['Home', 'L2/3', 'L4', 'L5']
            selected_options = st.multiselect('Select levels', options)
            p_title = ['Baseline', 'Intervention']
            col0, col1 = st.columns(2)
            col = [col0, col1]


            for i in range(2):
                if i == 0:
                    df = pd.DataFrame(MMR_b)
                else:
                    df = pd.DataFrame(MMR_i)

                df['month'] = range(1, 35)
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
                        title=p_title[i],
                    )
                    .mark_line()
                    .encode(
                        x=alt.X("month", axis=alt.Axis(title="Month")),
                        y=alt.Y("value", axis=alt.Axis(title="MMR"), scale=alt.Scale(domain=[0, 5])),
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

                with col[i]:
                    st.altair_chart(chart)

        if selected_plotB == "MMR":
            st.markdown("<h3 style='text-align: left;'>MMR of SDR subcounties</h3>", unsafe_allow_html=True)
            options = select_SCs #['Butere', 'Lugari', 'Malava']
            selected_options = st.selectbox('Select one SDR subcounty:', options)
            p_title = ['Baseline', 'Intervention']
            col0, col1 = st.columns(2)
            col = [col0, col1]
            def categorize_value(value):
                if value in options:
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
                df = df[df['Subcounty'] == selected_options]
                df = df.melt(id_vars=['Subcounty','SDR','Month'], var_name='level', value_name='value')
                chart = (
                        alt.Chart(
                            data=df,
                            title=p_title[i],
                        )
                        .mark_line()
                        .encode(
                            x=alt.X("Month", axis=alt.Axis(title="Month")),
                            y=alt.Y("value", axis=alt.Axis(title="MMR"),
                                    #scale=alt.Scale(domain=[0, 5])
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

                with col[i]:
                    st.altair_chart(chart)

            st.markdown("<h3 style='text-align: left;'>MMR of Non-SDR subcounties</h3>", unsafe_allow_html=True)
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
                df = df[df['SDR'] == 'Non-SDR']
                df = df.melt(id_vars=['Subcounty','SDR','Month'], var_name='level', value_name='value')
                df_summary = df.groupby(['Month', 'level'], as_index=False).mean()
                chart = (
                        alt.Chart(
                            data=df_summary,
                            title=p_title[i],
                        )
                        .mark_line()
                        .encode(
                            x=alt.X("Month", axis=alt.Axis(title="Month")),
                            y=alt.Y("value", axis=alt.Axis(title="MMR"), scale=alt.Scale(domain=[0, 5])),
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

                with col[i]:
                    st.altair_chart(chart)

        if selected_plotB == "Live births":
            st.markdown("<h3 style='text-align: left;'>%Live births of SDR subcounties</h3>", unsafe_allow_html=True)
            options = select_SCs #['Butere', 'Lugari', 'Malava']
            selected_options = st.selectbox('Select one SDR subcounty:', options)
            p_title = ['Baseline', 'Intervention']
            col0, col1 = st.columns(2)
            col = [col0, col1]


            def categorize_value(value):
                if value in options:
                    return 'SDR'
                else:
                    return 'Non-SDR'


            for i in range(2):
                if i == 0:
                    df = pd.DataFrame(bLB_SC)
                else:
                    df = pd.DataFrame(LB_SC)

                df.columns = ['Subcounty', 'Month', 'Home', 'L2/3', 'L4', 'L5', 'Sum']
                cols = df.columns[2:6]
                df[cols] = df[cols].div(df['Sum'], axis=0)
                df = df[['Subcounty', 'Month', 'Home', 'L2/3', 'L4', 'L5']]
                df['Subcounty'] = df['Subcounty'].replace(SC_ID)
                df['SDR'] = df['Subcounty'].apply(lambda x: categorize_value(x))
                df = df[df['Subcounty'] == selected_options]
                df = df.melt(id_vars=['Subcounty', 'SDR', 'Month'], var_name='level', value_name='value')

                chart = (
                    alt.Chart(
                        data=df,
                        title=p_title[i],
                    )
                    .mark_line()
                    .encode(
                        x=alt.X("Month", axis=alt.Axis(title="Month")),
                        y=alt.Y("value", axis=alt.Axis(title="%Live births"), scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("level:N").title("Level"),
                    ).properties(
                        width=500,
                        height=300
                    )
                )
                chart = chart.properties(
                ).configure_title(
                    anchor='middle'
                )

                with col[i]:
                    st.altair_chart(chart)

            st.markdown("<h3 style='text-align: left;'>%Live births of Non-SDR subcounties</h3>", unsafe_allow_html=True)
            col0, col1 = st.columns(2)
            col = [col0, col1]
            for i in range(2):
                if i == 0:
                    df = pd.DataFrame(bLB_SC)
                else:
                    df = pd.DataFrame(LB_SC)

                df.columns = ['Subcounty', 'Month', 'Home', 'L2/3', 'L4', 'L5', 'Sum']
                cols = df.columns[2:6]
                df[cols] = df[cols].div(df['Sum'], axis=0)
                df = df[['Subcounty', 'Month', 'Home', 'L2/3', 'L4', 'L5']]
                df['Subcounty'] = df['Subcounty'].replace(SC_ID)
                df['SDR'] = df['Subcounty'].apply(lambda x: categorize_value(x))
                df = df[df['SDR'] == 'Non-SDR']
                df = df.melt(id_vars=['Subcounty', 'SDR', 'Month'], var_name='level', value_name='value')

                df_summary = df.groupby(['Month', 'level'], as_index=False).mean()
                chart = (
                    alt.Chart(
                        data=df_summary,
                        title=p_title[i],
                    )
                    .mark_line()
                    .encode(
                        x=alt.X("Month", axis=alt.Axis(title="Month")),
                        y=alt.Y("value", axis=alt.Axis(title="%Live births"), scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("level:N").title("Level"),
                    ).properties(
                        width=500,
                        height=300
                    )
                )
                chart = chart.properties(
                ).configure_title(
                    anchor='middle'
                )

                with col[i]:
                    st.altair_chart(chart)

        if selected_plotB == "Complication rate":
            st.markdown("<h3 style='text-align: left;'>Complication rate (per 1000 live births) of SDR subcounties</h3>", unsafe_allow_html=True)
            options = select_SCs #['Butere', 'Lugari', 'Malava']
            selected_options = st.selectbox('Select one SDR subcounty:', options)
            p_title = ['Baseline', 'Intervention']
            col0, col1 = st.columns(2)
            col = [col0, col1]


            def categorize_value(value):
                if value in options:
                    return 'SDR'
                else:
                    return 'Non-SDR'


            for i in range(2):
                if i == 0:
                    df = pd.DataFrame(bComR_SC)
                else:
                    df = pd.DataFrame(ComR_SC)

                df.columns = ['Subcounty', 'Month', 'PPH', 'Sepsis', 'Eclampsia', 'Obstructed Labor', 'Others']
                df['Subcounty'] = df['Subcounty'].replace(SC_ID)
                df['SDR'] = df['Subcounty'].apply(lambda x: categorize_value(x))
                df = df[df['Subcounty'] == selected_options]
                df = df.melt(id_vars=['Subcounty', 'SDR', 'Month'], var_name='type', value_name='value')

                chart = (
                    alt.Chart(
                        data=df,
                        title=p_title[i],
                    )
                    .mark_line()
                    .encode(
                        x=alt.X("Month", axis=alt.Axis(title="Month")),
                        y=alt.Y("value", axis=alt.Axis(title="Complication rate"), scale=alt.Scale(domain=[0, 20])),
                        color=alt.Color("type:N", scale=alt.Scale(domain=['PPH', 'Sepsis', 'Eclampsia', 'Obstructed Labor', 'Others'])).title("Type"),
                    ).properties(
                        width=500,
                        height=300
                    )
                )
                chart = chart.properties(
                ).configure_title(
                    anchor='middle'
                )

                with col[i]:
                    st.altair_chart(chart)

            st.markdown("<h3 style='text-align: left;'>Complication rate (per 1000 live births) of Non-SDR subcounties</h3>", unsafe_allow_html=True)
            col0, col1 = st.columns(2)
            col = [col0, col1]
            for i in range(2):
                if i == 0:
                    df = pd.DataFrame(bComR_SC)
                else:
                    df = pd.DataFrame(ComR_SC)

                df.columns = ['Subcounty', 'Month', 'PPH', 'Sepsis', 'Eclampsia', 'Obstructed Labor', 'Others']
                df['Subcounty'] = df['Subcounty'].replace(SC_ID)
                df['SDR'] = df['Subcounty'].apply(lambda x: categorize_value(x))
                df = df[df['SDR'] == 'Non-SDR']
                df = df.melt(id_vars=['Subcounty', 'SDR', 'Month'], var_name='type', value_name='value')
                df_summary = df.groupby(['Month', 'type'], as_index=False).mean()

                chart = (
                    alt.Chart(
                        data=df_summary,
                        title=p_title[i],
                    )
                    .mark_line()
                    .encode(
                        x=alt.X("Month", axis=alt.Axis(title="Month")),
                        y=alt.Y("value", axis=alt.Axis(title="Complication rate"), scale=alt.Scale(domain=[0, 20])),
                        color=alt.Color("type:N", scale=alt.Scale(domain=['PPH', 'Sepsis', 'Eclampsia', 'Obstructed Labor', 'Others'])).title("Type"),
                    ).properties(
                        width=500,
                        height=300
                    )
                )
                chart = chart.properties(
                ).configure_title(
                    anchor='middle'
                )

                with col[i]:
                    st.altair_chart(chart)

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
