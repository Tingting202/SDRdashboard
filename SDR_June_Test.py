import streamlit as st
import numpy as np
import pydeck as pdk
import plotly.express as px
from SDR_Mother import Mother
from SDR_Facility import Facility
from SDR_choice import get_choices, get_probs
from SDR_opt import opt_anc
from itertools import islice

import pandas as pd
from matplotlib import pyplot as plt
import math
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import plotly.graph_objects as go

flag_initiate = False


@st.cache_data
def ready_data():
    # FOR OPTIMIZATION
    sheet_name = 'Sheet2'
    df_metrics = pd.read_excel('facility_data.xlsx', sheet_name=sheet_name)
    df_facility = pd.read_excel('Facility Key.xlsx')

    # FOR CHOICE MODEL
    df = pd.read_excel('Facility Key Categories.xlsx')  # get key with
    # added categorization
    df = df.iloc[0:399, :]
    df_ref = pd.read_excel("Referrals.xlsx", sheet_name='Simplified')  # get referral data
    df_ref.columns = ['Origin', 'Level_Origin', 'Dest', 'Level_Dest', 'Complication']
    referrals = df_ref
    referrals = referrals.loc[
        (referrals['Origin'] <= 401) & (referrals['Dest'] <= 401) & (pd.notna(referrals['Complication']))].reset_index(
        drop=True)
    df_data = pd.read_excel('SDR Consolidated Data 1.xlsx', sheet_name='summary_2022')
    df_births = df_data['live_birth']  # get live births
    df_ref = df_ref.groupby(['Origin', 'Dest']).size().reset_index(name='group_size')  # get unique referral counts

    # FOR OPTIMIZATION: minor cleaning and selection
    df_facility = df_facility.loc[(df_facility['Key'] <= 401) | (df_facility['Key'] == 430),]
    df_facility = df_facility.reset_index(drop=True)
    # origin = df_metrics.loc[df_metrics['ANC visit (New)'] > 0, ['Key', 'ANC visit (New)']]
    origin = df_data[['Key', 'anc_new']].loc[df_data['anc_new'] > 0, :]
    dest = df_metrics.loc[df_metrics['Live Births'] > 0, ['Key', 'Live Births']]
    df_facility['Location'] = df_facility.apply(lambda row: [row['Latitude'], row['Longitude']], axis=1)
    df_facility['Level Facility'] = df_facility['Level Facility'].replace(2, 3)
    new_row = {'Key': 430, 'Live Births': 100000}
    dest = dest.append(new_row, ignore_index=True)
    facilities = df_facility[['Key', 'Location']].set_index('Key')['Location']
    levels = df_facility[['Key', 'Level Facility']].set_index('Key')['Level Facility']
    c_section = df_metrics[['Key', 'C-Sections']].set_index('Key')['C-Sections']
    c_section[430] = 0

    # FOR CHOICE MODEL
    l1 = list(df['Key'][df['Categorization'] == 1])
    l2 = list(df['Key'][df['Categorization'] == 2])
    l3 = list(df['Key'][df['Categorization'] == 3])

    df_births = np.log10(df_births)
    df_births[np.isinf(df_births)] = 0
    df_ref['group_size'] = np.log10(df_ref['group_size'])
    # reshape latitude and longitude for distance calculation
    latlong = np.column_stack((df['Latitude'], df['Longitude']))
    # get distance, unique referral count between two facilities
    distance = np.zeros((len(df['Key']), len(df['Key'])))
    ref_count = np.zeros((len(df['Key']), len(df['Key'])))
    for i in range(len(df['Key'])):
        for j in range(len(df['Key'])):
            distance[i, j] = np.linalg.norm(latlong[i] - latlong[j])
            ref_len = len(df_ref['group_size'][(df_ref['Origin'] == df['Key'][i]) & (df_ref['Dest'] == df['Key'][j])])
            if ref_len > 0:
                ref_count[i, j] = df_ref['group_size'][
                    (df_ref['Origin'] == df['Key'][i]) & (df_ref['Dest'] == df['Key'][j])]

    return df, df_facility, df_data, origin, dest, facilities, levels, c_section, l1, l2, l3, \
        df_births, df_ref, distance, ref_count


[df, df_facility, df_data, origin, dest, facilities, levels, c_section,
 l1, l2, l3, df_births, df_ref, distance, ref_count] = ready_data()

# DEFINE COEFFICIENTS
# for OPTIMIZATION
betas = [2, -1.5, -0.5]

# MODEL 4
coef = np.matrix([
    [0, -3.8481, 11.16878],  # intercept
    [0, 0, 0],  # births
    [0, 14.989, -1.203],  # original category 3
    [0, 17.45657, -0.54457],  # original category 4
    [0, 0, 0],  # complication 1, choice 1-3
    [0, 0, 0],  # complication 2, choices 1-3
    [1.1021, -2.422, -0.9806],  # distances, choices 1-3
    [0, 0, 0],  # referral count, choices 1-3
])

# FOR CHOICE
# initialize facilities
facility_objects = []
for key, facility in df_facility.iterrows():
    facility_objects.append(Facility(facility[0], facility[2], facility[8]))

# initialize mothers
mothers = []
risk = [0, 1, 2]
for number, facility in zip(origin['anc_new'], origin['Key']):
    distribution = np.random.multinomial(number, [0.33, 0.26, 0.41], 1).flatten()
    facility_object = [object for object in facility_objects if object.idx == facility][0]
    for i in range(3):
        for person in range(distribution[i]):
            mothers.append(Mother(facility_object, risk[i]))


@st.cache_data
def cost_m(facilities, df_data):
    # GET COST MATRICES
    distance_opt = np.zeros((len(facilities), len(facilities), 3))
    distance_d = np.matrix(
        [math.dist(facilities[x], facilities[y]) for x in facilities.index for y in facilities.index])
    distance_d = np.reshape(distance_d, (len(facilities), len(facilities)))
    for i in range(3):
        distance_opt[:, :, i] = distance_d
    distance_opt[np.isnan(distance_opt)] = 100
    # mismatch level across each risk level
    mismatch = np.zeros((len(facilities), len(facilities), 3))
    for x in range(len(facilities)):
        for y in range(len(facilities)):
            for z in range(3):
                mismatch[x, y, z] = levels.to_list()[y] - z - 2
    # csect_opt
    csect_opt = df_data['c_section']
    csect_opt[399] = 0
    csect = np.zeros((len(facilities), len(facilities), 3))
    for x in range(len(facilities)):
        for y in range(len(facilities)):
            for z in range(3):
                csect[x, y, z] = csect_opt[x]

    cost = 2 * distance_opt - 1.5 * mismatch - 0.5 * csect
    cost_matrix = cost.reshape(1, len(facilities) ** 2 * 3)
    return cost_matrix


cost_matrix = cost_m(facilities, df_data)

key_solution = opt_anc(facilities, mothers, df_data, cost_matrix)

facility_obj_dict = {object.idx: object for object in facility_objects}
for key, value in key_solution.items():
    if value > 0:
        key_0 = df_facility['Key'][key[0]]
        key_1 = df_facility['Key'][key[1]]
        match_anc = (mother for mother in mothers if
                     mother.facility_anc.idx == key_0 and mother.risk == key[2] and mother.facility_delivery is None)
        object = facility_obj_dict[key_1]
        for mother in islice(match_anc, int(value)):
            mother.facility_delivery = object
            object.facility_deliveries += 1

t_p = 0.5
for index, mother in enumerate(mothers):
    mother.develop_complication()
    mother.choose_refer(t_p)
    if (mother.transfer == 1) & (mother.facility_delivery.idx != 430):
        mother.choice_referral(get_choices, get_probs, coef, facility_objects, l1, l2, df, distance, ref_count,
                               df_births)

    else:
        mother.adjust_final()
    if (mother.transfer == 1) & (mother.facility_delivery.idx == 430):
        mother.final_deliver = 'Unknown'
# CALCULATE MORTALITY
for index, facility in enumerate(facility_objects):
    facility.mortality()

# COLLECT THE MOTHERS DATA
facility_anc = []
facility_final = []
facility_initial = []
complications = []
risks = []
facility_delivery = []
transfer = []

for mother in mothers:
    if mother.final_deliver != 'Unknown':
        facility_final.append(mother.final_deliver.idx)
    else:
        facility_final.append('Unknown')
    facility_anc.append(mother.facility_anc.idx)
    risks.append(mother.risk)
    facility_delivery.append(mother.facility_delivery.idx)
    facility_initial.append(mother.facility_delivery.idx)
    complications.append(mother.complication_level)
    transfer.append(mother.transfer)

df_mother = pd.DataFrame({'ANC': facility_anc, 'Complications': complications, 'Delivery - Initial': facility_initial,
                          'Delivery - Final': facility_final, 'Transfer': transfer, 'Risk': risks,
                          'Delivery': facility_delivery})

# COLLECT THE FACILITY DATA
f_transfers = []
f_complications_high = []
f_complications_low = []
f_mortalities = []
f_key = []

for facility in facility_objects:
    f_transfers.append(facility.transferred)
    f_complications_high.append(facility.complication_high)
    f_complications_low.append(facility.complication_low)
    f_mortalities.append(facility.count)
    f_key.append(facility.idx)

df_fo = pd.DataFrame({'Key': f_key, 'Transferred': f_transfers, 'Complications High': f_complications_high,
                      'Complications Low': f_complications_low,
                      'Mortalities': f_mortalities})

refs = df_mother.loc[:, ['Delivery - Initial', 'Delivery - Final']].groupby(['Delivery - Initial',
                                                                             'Delivery - Final']).value_counts().reset_index(
    name='Counts')
refs = refs.loc[refs['Delivery - Initial'] != refs['Delivery - Final'], :]
refs = refs.loc[refs['Delivery - Final'] != 'Unknown', :]
refs = refs.merge(df_facility[['Key', 'Level Facility']],
                  left_on='Delivery - Initial', right_on='Key').merge(df_facility[['Key', 'Level Facility']],
                                                                      left_on='Delivery - Final', right_on='Key').drop(
    ['Key_x', 'Key_y'], axis=1)


# plot referrals between specific facilities
def ref_plot(refs, level_origin, level_dest, traces):
    ref_data = {}
    latitude, longitude, linewidth = [], [], []
    lines = []

    refs = refs.loc[(refs['Level Facility_x'] == level_origin) & (refs['Level Facility_y'] == level_dest), :]

    for x, line in refs.iterrows():
        lines.append(str(x))
        latitude.append([facilities[line[0]][0], facilities[line[1]][0]])
        longitude.append([facilities[line[0]][1], facilities[line[1]][1]])
        linewidth.append(line[2])
        # linewidth.append(line[2])
    ref_data['Latitude'] = latitude
    ref_data['Longitude'] = longitude
    ref_data['Line'] = lines
    ref_data['Width'] = linewidth

    df = pd.DataFrame(ref_data)
    #
    for _, row in df.iterrows():
        trace = go.Scattermapbox(
            lat=row['Latitude'],
            lon=row['Longitude'],
            mode='lines',
            line=dict(width=row['Width'] / 3),
            name=row['Line'],
            marker=dict(color='blue')
        )
        traces.append(trace)

    lengths = df.shape[0]

    return traces, lengths


### START PLOTTING HERE ###

midpoint = (np.mean(df_facility['Latitude']), np.mean(df_facility['Longitude']))
category_list = df_facility['Level Facility'].to_list()
color_mapping = {
    0: 'blue',
    2: 'blue',
    3: 'blue',
    4: 'red',
    5: 'black'
    # Add more category-color mappings as needed
}
color_list = [color_mapping.get(category, 'blue') for category in category_list]

layout = go.Layout(
    mapbox=dict(
        style='open-street-map',
        zoom=8,
        center=dict(lat=midpoint[0], lon=midpoint[1]),
    ),
)

### PLOT 1
facilitytrace = go.Scattermapbox(
    lat=df_facility['Latitude'],
    lon=df_facility['Longitude'],
    text=df_facility['organisationunitname'],
    marker=dict(
        size=7,
        color=color_list
    ))

fig3 = go.Figure(data=facilitytrace, layout=layout)
fig3.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.write('Facilities in Kakamega County')
st.plotly_chart(fig3)

### PLOT 2
initial_delivery_distribution = df_mother.loc[:, ['ANC', 'Delivery - Initial']].groupby(
    ['ANC', 'Delivery - Initial']).value_counts().reset_index(name='Counts')

anc_data = {}
latitude, longitude, linewidth = [], [], []
lines = []

for x, line in initial_delivery_distribution.iterrows():
    lines.append(str(x))
    latitude.append([facilities[line[0]][0], facilities[line[1]][0]])
    longitude.append([facilities[line[0]][1], facilities[line[1]][1]])
    linewidth.append(line[2])

anc_data['Latitude'] = latitude
anc_data['Longitude'] = longitude
anc_data['Line'] = lines
anc_data['Width'] = linewidth

df = pd.DataFrame(anc_data)
traces2 = []
for _, row in df.iterrows():
    trace = go.Scattermapbox(
        lat=row['Latitude'],
        lon=row['Longitude'],
        mode='lines',
        line=dict(width=row['Width'] / 75),
        name=row['Line'],
        marker=dict(color='blue')
    )
    traces2.append(trace)

traces2.append(facilitytrace)

traces2 = [trace.update(showlegend=False) for trace in traces2]
fig2 = go.Figure(data=traces2, layout=layout)
fig2.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.write('Initial Distribution of Mothers to Delivery Facilities')
st.plotly_chart(fig2)

### PLOT 3
traces = []
traces.append(facilitytrace)

layout = go.Layout(
    mapbox=dict(
        style='open-street-map',
        zoom=8,
        center=dict(lat=midpoint[0], lon=midpoint[1]),
    ),
)
traces, length1 = ref_plot(refs, 3, 4, traces)
traces, length2 = ref_plot(refs, 3, 5, traces)
traces, length3 = ref_plot(refs, 4, 4, traces)
traces, length4 = ref_plot(refs, 4, 5, traces)

fig = go.Figure(data=traces, layout=layout)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([dict(label="L3 to L4",
                               method="update",
                               args=[{"visible": [True] + [True] * length1 + [False] * length2 +
                                                 [False] * length3 + [False] * length4},
                                     {"title": "Referrals from L3 to L4",
                                      "showlegend": False}]),
                          dict(label="L3 to L5",
                               method="update",
                               args=[{"visible": [True] + [False] * length1 + [True] * length2 +
                                                 [False] * length3 + [False] * length4},
                                     {"title": "Referrals from L3 to L5"}]),
                          dict(label="L4 to L4",
                               method="update",
                               args=[{"visible": [True] + [False] * length1 + [False] * length2 +
                                                 [True] * length3 + [False] * length4},
                                     {"title": "Referrals from L4 to L4"}]),
                          dict(label="L4 to L5",
                               method="update",
                               args=[{"visible": [True] + [False] * length1 + [False] * length2 +
                                                 [False] * length3 + [True] * length4},
                                     {"title": "Referrals from L4 to L5"}]),
                          ]),
        )
    ])

fig.update_layout(title_text="Referrals")

st.write('Referrals')
st.plotly_chart(fig)
