import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd
import altair as alt


st.set_page_config(layout="wide", page_title="ABM Version 1 Simulation Results", page_icon="📈")
st.sidebar.header("ABM Version 1 Simulation Results")

# SETTING UP INPUTS

# POPULATION, FACILITY INFORMATION
total_births = 66366
facility_births = 43138
births_p = [0.35, 0.26, 0.31, 0.08]  # Home, L2/3, L4, L5
home_births = total_births * births_p[0]
# l4_5_ratio = births_p[2]/births_p[3]
l4_5_ratio = 4

# SCENARIOS, ADJUSTED FOR L4/L5
conservative = [0.35, 0.35, 0.35, 0.33, 0.30,
                0.20, 0.20, 0.18, 0.17, 0.17,
                0.45, 0.45, 0.47, 0.50, 0.53]
moderate = [0.35, 0.35, 0.30, 0.25, 0.20,
            0.20, 0.18, 0.18, 0.15, 0.12,
            0.45, 0.47, 0.52, 0.60, 0.68]
aggressive = [0.35, 0.35, 0.30, 0.20, 0.05,
              0.20, 0.16, 0.10, 0.04, 0.05,
              0.45, 0.49, 0.60, 0.76, 0.90]

no_sdr = np.array([[0.35, 0.35, 0.35, 0.35, 0.35],
                   [0.2, 0.2, 0.2, 0.2, 0.2],
                   [0.3375, 0.3375, 0.3375, 0.3375, 0.3375],
                   [0.1125, 0.1125, 0.1125, 0.1125, 0.1125]])

scenario_names = ["conservative", "moderate", "aggressive"]
scenarios = [conservative, moderate, aggressive]

for scenario, scenario_name in zip(scenarios, scenario_names):
    scenario = np.array(scenario)
    scenario = np.reshape(scenario, (3, 5))
    l5 = scenario[2, :] / l4_5_ratio
    l4 = scenario[2, :] - l5
    scenario = np.delete(scenario, 2, 0)
    scenario = np.vstack([scenario, l4, l5])
    globals()[scenario_name] = scenario


def calculate_scenario(scenario, mother, anc, sdr):
    anc_effects = [0, 0.1, 0.2]  # reduction in complications (different levels of ANC) (reduces complications)
    sdr_effects = [0, 0.1, 0.2]  # reduction in mortality:complications rate (different levels SDR) (reduces death)

    # COMPLICATION RATES AT EACH FACILITY LEVEL
    c_pct = np.repeat(0.035, 5)
    # for year in range(1,6):
    # c_pct[year-1] = c_pct[year-1]*(1-anc_effects[anc]*(year-1)/4)
    c_high_proportion = 25 / 100
    c_low_pct = c_pct * (1 - c_high_proportion)
    c_high_pct = c_pct * (c_high_proportion)

    # out of high and low complications
    C_transfer_high_home_L5 = 0.10  # # percentage of home with high complications transferred to L5
    C_transfer_high_home_L4 = 0.21  # # percentage of home with high complications transferred to L4
    C_transfer_low_home_L4 = 0.06  #
    C_transfer_low_home_L23 = 0.10  # #

    C_transfer_high_L23_L5 = 0.67  #
    C_transfer_high_L23_L4 = 0.33  #
    C_transfer_low_L23_L4 = 0.50  #

    C_transfer_high_L4_L5 = 0.41
    C_transfer_low_L4_L5 = 0.21

    f_scenario = scenario

    scenario_low = f_scenario * c_low_pct
    scenario_high = f_scenario * c_high_pct

    C_home_low_in = np.array([0, 0, 0, 0, 0])
    C_home_low_out = (C_transfer_low_home_L23 + C_transfer_low_home_L4) * (scenario_low[0])
    C_home_low_change = C_home_low_in - C_home_low_out
    C_home_high_in = np.array([0, 0, 0, 0, 0])
    C_home_high_out = (C_transfer_high_home_L5 + C_transfer_high_home_L4) * (scenario_high[0])
    C_home_high_change = C_home_high_in - C_home_high_out

    C_l23_low_in = C_transfer_low_home_L23 * (scenario_low[0])
    C_l23_low_out = (C_transfer_low_L23_L4) * (scenario_low[1])
    C_l23_low_change = C_l23_low_in - C_l23_low_out
    C_l23_high_in = np.array([0, 0, 0, 0, 0])
    C_l23_high_out = (C_transfer_high_L23_L4 + C_transfer_high_L23_L5) * (scenario_high[1])
    C_l23_high_change = C_l23_high_in - C_l23_high_out

    C_l4_low_in = C_transfer_low_home_L4 * (scenario_low[0]) + C_transfer_low_L23_L4 * (scenario_low[1])
    C_l4_low_out = C_transfer_low_L4_L5 * (scenario_low[2])
    C_l4_low_change = C_l4_low_in - C_l4_low_out
    C_l4_high_in = C_transfer_high_home_L4 * (scenario_high[0]) + C_transfer_high_L23_L4 * (scenario_high[1])
    C_l4_high_out = C_transfer_high_L4_L5 * (scenario_high[2])
    C_l4_high_change = C_l4_high_in - C_l4_high_out

    C_l5_low_in = C_transfer_low_L4_L5 * (scenario_low[2])
    C_l5_low_out = np.array([0, 0, 0, 0, 0])
    C_l5_low_change = C_l5_low_in - C_l5_low_out
    C_l5_high_in = C_transfer_high_home_L5 * (scenario_high[0]) + C_transfer_high_L23_L5 * (scenario_high[1]) + \
                   C_transfer_high_L4_L5 * (scenario_high[2])
    C_l5_high_out = np.array([0, 0, 0, 0, 0])
    C_l5_high_change = C_l5_high_in - C_l5_high_out

    scenario_transferin_low = np.array([
        C_home_low_in,
        C_l23_low_in,
        C_l4_low_in,
        C_l5_low_in
    ])
    scenario_transferin_high = np.array([
        C_home_high_in,
        C_l23_high_in,
        C_l4_high_in,
        C_l5_high_in
    ])
    scenario_notransfer_low = np.array([
        scenario_low[0] - C_home_low_out,
        scenario_low[1] - C_l23_low_out,
        scenario_low[2] - C_l4_low_out,
        scenario_low[3] - C_l5_low_out
    ])

    scenario_notransfer_high = np.array([
        scenario_high[0] - C_home_high_out,
        scenario_high[1] - C_l23_high_out,
        scenario_high[2] - C_l4_high_out,
        scenario_high[3] - C_l5_high_out
    ])

    complications_post = np.array([
        c_pct * scenario[0] + C_home_low_change + C_home_high_change,
        c_pct * scenario[1] + C_l23_low_change + C_l23_high_change,
        c_pct * scenario[2] + C_l4_low_change + C_l4_high_change,
        c_pct * scenario[3] + C_l5_low_change + C_l5_high_change
    ])

    scenario_post = np.array([
        scenario[0] + C_home_low_change + C_home_high_change,
        scenario[1] + C_l23_low_change + C_l23_high_change,
        scenario[2] + C_l4_low_change + C_l4_high_change,
        scenario[3] + C_l5_low_change + C_l5_high_change
    ])

    ### MORTALITY CALCULATIONS
    if mother == 1:
        # MATERNAL
        rho4 = 0.3
        rho5 = rho4 ** 2

        D_low = np.array([0.065, 0.01, 0.01 * rho4, 0.01 * rho5])
        D_high = np.array([1, 0.15, 0.049, 0.021])
        D_low = np.reshape(np.tile(D_low, 5), (5, 4)).T
        D_high = np.reshape(np.tile(D_high, 5), (5, 4)).T

        for year in range(1, 6):
            D_low[:, year - 1] = D_low[:, year - 1] * (1 - sdr_effects[sdr] * (year - 1) / 4)
            D_high[:, year - 1] = D_high[:, year - 1] * (1 - sdr_effects[sdr] * (year - 1) / 4)

            D_low[:, year - 1] = D_low[:, year - 1] - 10 * anc_effects[anc] * (year - 1) * 0.00018  # # # Newly added
            D_high[:, year - 1] = D_high[:, year - 1] - 10 * anc_effects[anc] * (year - 1) * 0.00018  # # # Newly added

            D_low[:, year - 1] = D_low[:, year - 1] - (year - 1) * 0.0003
            D_high[:, year - 1] = D_high[:, year - 1] - (year - 1) * 0.0003

            D_low[D_low < 0] = 0
            D_high[D_high < 0] = 0

        mult_D = 3

    if mother == 0:
        # NEONATAL
        rho4 = 0.7
        rho5 = rho4 ** 2

        D_low = np.array([0.7, 0.1, 0.1 * rho4, 0.1 * rho5])
        D_high = np.array([1, 1, rho4, 0.5])
        D_low = np.reshape(np.tile(D_low, 5), (5, 4)).T
        D_high = np.reshape(np.tile(D_high, 5), (5, 4)).T

        for year in range(1, 6):
            D_low[:, year - 1] = D_low[:, year - 1] * (1 - sdr_effects[sdr] * (year - 1) / 4)
            D_high[:, year - 1] = D_high[:, year - 1] * (1 - sdr_effects[sdr] * (year - 1) / 4)

            D_low[:, year - 1] = D_low[:, year - 1] - (year - 1) * 0.0003
            D_high[:, year - 1] = D_high[:, year - 1] - (year - 1) * 0.0003

            D_low[D_low < 0] = 0
            D_high[D_high < 0] = 0

        mult_D = 2

    D_low_mult = D_low * mult_D
    D_high_mult = D_high * mult_D

    d_transferin_low = D_low_mult * scenario_transferin_low
    d_transferin_high = D_high_mult * scenario_transferin_high
    d_notransfer_low = D_low * scenario_notransfer_low
    d_notransfer_high = D_high * scenario_notransfer_high

    ### SUMMARIZATIONS AND PLOTTING
    total_deaths = (d_transferin_low + d_transferin_high + d_notransfer_low + d_notransfer_high) * total_births
    total_complications = complications_post * total_births
    total_pretransfer = scenario
    total_posttransfer = scenario_post

    return (total_deaths, total_complications, total_pretransfer, total_posttransfer)


### Plotting function
def scenario_plot(tickers, names, ylabel):
    hold_fig = []
    for n, ticker in enumerate(tickers):
        ticker = pd.DataFrame(ticker.T)
        x = ['2021', '2022', '2023', '2024', '2025']
        ticker.columns = ['Home', 'L2/3', 'L4', 'L5']
        ticker['x'] = x
        ticker_m = ticker.melt(id_vars=['x'], value_vars=['Home', 'L2/3', 'L4', 'L5'])
        hold_fig.append(alt.Chart(ticker_m, title=names[n]).mark_bar().encode(
            color='variable',
            x=alt.X('x', title='Years'),
            y=alt.Y('sum(value)', title=ylabel),
        ))

    left,middle,right = st.columns(3)
    with left:
        hold_fig[0]
    with middle:
        hold_fig[1]
    with right:
        hold_fig[2]

def scenario_table(tickers):
    daly = []
    for n, ticker in enumerate(tickers):
        ticker = pd.DataFrame(ticker.T)
        daly.append(int((ticker.iloc[0,0:4].sum()*(37.4 + 21.77*12.35)) - (ticker.iloc[4,0:4].sum()*(37.4 + 21.77*12.35))))
    daly = pd.DataFrame(daly)
    daly['costs'] = [2300040000, 2049280000, 1919570000]
    daly['cost per daly'] = daly['costs'] / daly[0]
    daly = daly.loc[:, [0, 'cost per daly']]
    daly['cost per daly'] = [int(c) for c in daly['cost per daly']]
    daly.columns = ['DALYs averted', 'Costs per DALY averted']
    daly.index = ['Conservative', 'Moderate', 'Aggressive']
    left, middle, right = st.columns(3)
    with left:
        st.dataframe(daly, use_container_width=True)

def scenario_input(a,b,c,d, l4_5_ratio):
    scenario = [(1-a-b), 0, 0, 0, (1-c-d),
                a, 0, 0, 0, c,
                b, 0, 0, 0, d]
    scenario = np.array(scenario)
    scenario = np.reshape(scenario, (3, 5))
    l5 = scenario[2, :] / l4_5_ratio
    l4 = scenario[2, :] - l5
    scenario = np.delete(scenario, 2, 0)
    scenario = np.vstack([scenario, l4, l5])
    return scenario

def scenario_input_plot(scenario, mother, anc, sdr):
    ticker, p1, p2, p3 = calculate_scenario(scenario, mother, anc, sdr)
    ticker = pd.DataFrame(ticker.T)
    x = ['Baseline', '2022', '2023', '2024', 'Projected']
    ticker.columns = ['Home', 'L2/3', 'L4', 'L5']
    ticker['x'] = x
    ticker = ticker.iloc[[0,4],:]
    daly = int((ticker.iloc[0,0:4].sum()*(37.4+21.77*12.35)) - (ticker.iloc[1,0:4].sum()*(37.4+21.77*12.35)))
    ticker_m = ticker.melt(id_vars=['x'], value_vars=['Home', 'L2/3', 'L4', 'L5'])
    fig = alt.Chart(ticker_m).mark_bar().encode(
        color='variable',
        x=alt.X('x', title=''),
        y=alt.Y('sum(value)', title=ylabel),
    )
    left, middle, right = st.columns([1,5,1])
    with middle:
        st.altair_chart(fig, use_container_width=True)
        st.text(f'DALYs averted over four years: {daly}')

tickers = [conservative, moderate, aggressive]
names = ["conservative", "moderate", "aggressive"]
title = 'Proportion of Deliveries'
ylabel = 'Proportion of Deliveries'
st.subheader(title)
scenario_plot(tickers, names, ylabel)

st.subheader('Scenario Selection')
option = st.selectbox("Scenarios:", ["Projected Shift in Deliveries",
                                     "Moderate SDR Strategy with ANC effect",
                                     "Moderate SDR Strategy with ANC and SDR effect"])

if option == "Projected Shift in Deliveries":
    total_deaths1, total_complications, total_pretransfer, total_posttransfer = calculate_scenario(conservative, 1, 0,                                                                               0)
    total_deaths2, total_complications, total_pretransfer, total_posttransfer = calculate_scenario(moderate, 1, 0, 0)
    total_deaths3, total_complications, total_pretransfer, total_posttransfer = calculate_scenario(aggressive, 1, 0, 0)

    tickers = [total_deaths1, total_deaths2, total_deaths3]
    names = ["conservative", "moderate", "aggressive"]
    title = 'Number of Maternal Mortalities'
    ylabel = 'Maternal Mortalities'

    scenario_plot(tickers, names, ylabel)
    scenario_table(tickers)

elif option == "Moderate SDR Strategy with ANC effect":
    total_deaths1, total_complications, total_pretransfer, total_posttransfer = calculate_scenario(moderate, 1, 0, 0)
    total_deaths2, total_complications, total_pretransfer, total_posttransfer = calculate_scenario(moderate, 1, 1, 0)
    total_deaths3, total_complications, total_pretransfer, total_posttransfer = calculate_scenario(moderate, 1, 2, 0)

    tickers = [total_deaths1, total_deaths2, total_deaths3]
    names = ['as is', '10% Reduction', '20% Reduction']
    title = 'Number of Maternal Mortalities - ANC effect'
    ylabel = 'Maternal Mortalities'

    scenario_plot(tickers, names, ylabel)
    scenario_table(tickers)

elif option == 'Moderate SDR Strategy with ANC and SDR effect':
### Maternal Mortalities for Moderate SDR Strategy with Increased ANC Visits and SDR Impacts on Health Services Quality
    total_deaths1, total_complications, total_pretransfer, total_posttransfer = calculate_scenario(moderate, 1, 0, 0)
    total_deaths2, total_complications, total_pretransfer, total_posttransfer = calculate_scenario(moderate, 1, 1, 1)
    total_deaths3, total_complications, total_pretransfer, total_posttransfer = calculate_scenario(moderate, 1, 2, 2)

    tickers = [total_deaths1, total_deaths2, total_deaths3]
    names = ['as is', '10% Reduction', '20% Reduction']
    title = 'Number of Maternal Mortalities - ANC and SDR effect'
    ylabel = 'Maternal Mortalities'

    scenario_plot(tickers, names, ylabel)
    scenario_table(tickers)

st.subheader('Scenario Projection')
with st.form('Test'):
    left, right = st.columns(2)
    with left:
        b_45 = st.slider('Baseline: % Deliveries at L4/5', 0.0, 1.0, 0.05)
        b_23 = st.slider('Baseline: % Deliveries at L2/3', 0.0, 1.0, 0.05)
        anc = st.slider('ANC Effect:', 0, 2, 1)
    with right:
        p_45 = st.slider('Projected: % Deliveries at L4/5', 0.0, 1.0, 0.05)
        p_23 = st.slider('Projected: % Deliveries at L2/3', 0.0, 1.0, 0.05)
        sdr = st.slider('Quality of Care Effect:', 0, 2, 1)

    submitted = st.form_submit_button("Run Model")
    if submitted:
        scenario = scenario_input(b_23, b_45, p_23, p_45, 4)
        scenario_input_plot(scenario, 1, anc, sdr)
