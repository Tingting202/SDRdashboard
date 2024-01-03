import numpy as np
from sympy import symbols, Eq, solve
import pandas as pd
from scipy.optimize import fsolve

n_months = 36
t = np.arange(n_months)

Capacity = np.array([3972, 3198, 1020, 3168, 2412, 6798, 2580, 2136, 936, 2772, 1524, 4668])

sc_time ={
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
    }
}

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
])/100

sc_ID = {
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

sc = pd.DataFrame({
    'LB': sc_LB,
    'ANC': sc_ANC,
    'CLASS': sc_CLASS,
    'knowledge': sc_knowledge
})

i_scenarios = pd.DataFrame({
    'Supplies': i_supplies
})

for i in range(sc_time['n']):
    sc_time['LB1s'][i][0,:] = sc['LB'][i]

Capacity_ratio = np.zeros((n_months,sc_time['n']))
Push_back = np.zeros((n_months,sc_time['n']))
opinion = np.zeros((n_months, sc_time['n']))

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
])/100

m_transfer=2

f_mort_rates = np.array([
    [0.10, 25.00],
    [0.10, 25.00],
    [0.10, 6.00],
    [0.10, 3.00]
])/100
f_mort_rates = np.c_[f_mort_rates, f_mort_rates[:,1]*m_transfer]

# PPH, Sepsis, Eclampsia, Obstructed Labor
comp_transfer_props = np.array([0.2590361446, 0.1204819277, 0.2108433735, 0.4096385542])

b_comp_mort_rates = np.array([0.07058355, 0.1855773323, 0.3285932084, 0.02096039958, 0.3942855098])

p_ol = 0.01
p_other = 0.0044
p_severe = 0.358

def reset_INT():
    INTp = {
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

    return INTp

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
            df_aggregate.loc[i,j] = np.sum(np.array(df.loc[(slice(None), i), j]), axis=0)
    
    return df_aggregate

def community_effect(x):
    if x-0.5<0:
        x = -(1-x)
    else:
        x = x
    return 0.000015*(np.exp(10*x) - np.exp(-10*x))

def set_flags(subcounty, flags, i_scenarios, sc, global_vars):

    Capacity = np.array([3972, 3198, 1020, 3168, 2412, 6798, 2580, 2136, 936, 2772, 1524, 4668])
    
    flag_CHV = flags[0]
    flag_ANC = flags[1]
    flag_ANC_time = flags[2]
    flag_refer = flags[3]
    flag_trans = flags[4]
    flag_int1 = flags[5]   # Obstetric drape
    flag_int2 = flags[6]   # Anemia reduction through IV Iron
    flag_int3 = flags[7]   # ultrasound
    flag_int5 = flags[8]   # MgSO4 for eclampsia 
    flag_int6 = flags[9]   # antibiotics for maternal sepsis
    flag_sdr = flags[10]   # SDR
    flag_capacity = flags[11]
    flag_community = flags[12]

    INTp = reset_INT()

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

    for i,x in zip(range(sc_time['n']), subcounty):
            knowl = np.array([k for k in knowledge])
            suppl = np.array([s for s in supplies])
            if x==1:
                #effect = 0.2
                int_suppl =  np.array([max((1 + supply_added) * s, 1) for s in supplies])  #(1 + effect)*suppl
                if flag_sdr:
                    #effect = 0.2
                    knowl = np.array([max((1 + know_added) * k, 1) for k in knowledge]) #np.array([((1+effect) * k) for k in knowledge])
                    suppl = np.array([max((1 + supply_added) * s, 1) for s in supplies]) #np.array([((1+effect) * s) for s in supplies])
                    flag_CHV = 1
                    flag_int1 = 1
                    flag_int2 = 1
                    flag_int5 = 1
                    flag_int6 = 1
                    flag_capacity = 1
                if flag_CHV:
                    INTp['CHV']['sdr_effect'][i,:] = np.repeat(CHV_45 * CHV_cov, n_months)
                if flag_ANC: 
                    INTp['ANC']['effect'][i,:] = np.repeat(1 + ANCadded, n_months)
                if flag_ANC_time: 
                    INTp['ANC']['effect'][i,:] = 1 + np.linspace(0, ANCadded, n_months)
                if flag_refer:
                    INTp['refer']['effect'][i,:] = np.repeat(1 + referadded, n_months)
                if flag_trans: 
                    INTp['transfer']['effect'][i,:] = 1 + transadded
                if flag_int1: 
                    quality_factor = knowl[i]*int_suppl[2] - knowl[i]*suppl[2]
                    INTp['int1']['effect'][i,:] = 1- ((1-np.repeat(0.6, n_months)) * quality_factor)
                    INTp['int1']['coverage'][i,:] = quality_factor
                if flag_int2: 
                    quality_factor = knowl[i]*int_suppl[0] - knowl[i]*suppl[0]
                    INTp['int2']['effect'] = 1-((1-0.7) * quality_factor)
                    INTp['int2']['coverage'] = quality_factor
                if flag_int5: 
                    quality_factor =  knowl[i]*int_suppl[3] - knowl[i]*suppl[3]
                    INTp['int5']['effect'][i,:] = 1- ((1-np.repeat(0.59, n_months)) * quality_factor)
                    INTp['int5']['coverage'][i,:] = quality_factor 
                if flag_int6: 
                    quality_factor = knowl[i]*int_suppl[4] - knowl[i]*suppl[4]
                    INTp['int6']['effect'][i,:] = 1- ((1-np.repeat(0.8, n_months)) * quality_factor)
                    INTp['int6']['coverage'][i,:] = quality_factor
                if flag_capacity:
                    Capacity[i] = Capacity[i] * (1 + capacity_added)
                if flag_community: 
                    INTp['Community']['effect'][i] = 1
                    
            else: 
                quality_factor = knowl[i] * suppl[2]
                INTp['int1']['coverage'][i,:] = quality_factor
                quality_factor = knowl[i] * suppl[0]
                INTp['int2']['coverage'] = quality_factor
                quality_factor = knowl[i] * suppl[3]
                INTp['int5']['coverage'][i,:] = quality_factor
                quality_factor = knowl[i] * suppl[4]
                INTp['int6']['coverage'][i,:] = quality_factor

                INTp['CHV']['sdr_effect'][i,:] = np.repeat(0, n_months)
                INTp['ANC']['effect'][i,:] = np.repeat(1, n_months)
                INTp['refer']['effect'][i,:] = np.repeat(1, n_months) # transfer intervention cannot be subcounty specific
                # INT['transfer']['effect'] = np.ones((12, n_months))
                INTp['int1']['effect'][i,:] = np.repeat(1, n_months)
                # INT['int2']['effect'] = 1                          # intervention 2 cannot be subcounty specific 
                INTp['int5']['effect'][i,:] = np.repeat(1, n_months)
                INTp['int6']['effect'][i,:] = np.repeat(1, n_months)
                INTp['Community']['effect'][i] = 0
            
    i_INT = INTp
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
    flag_int1 = flags[5]   # Obstetric drape
    flag_int2 = flags[6]   # Anemia reduction through IV Iron
    flag_int3 = flags[7]   # ultrasound
    flag_int5 = flags[8]   # MgSO4 for eclampsia 
    flag_int6 = flags[9]   # antibiotics for maternal sepsis
    flag_sdr = flags[10]   # SDR
    flag_capacity = flags[11]
    flag_community = flags[12]

    #effect = 0.2
    # i_INT = set_flags(subcounty, flags, i_scenarios, sc)
    supply_added = global_vars[5]

    low_pph = np.sum(b_df_aggregate.loc[timepoint, 'Complications-Health'], axis=1)[0]*np.array([1-p_severe, p_severe])[0]
    high_pph = np.sum(b_df_aggregate.loc[timepoint, 'Complications-Health'], axis=1)[0]*np.array([1-p_severe, p_severe])[1]
    maternal_deaths = np.sum(b_df_aggregate.loc[timepoint, 'Deaths'])

    b_DALYs = DALYs[0]*low_pph*62.63 + DALYs[1]*high_pph*62.63 + DALYs[2]*maternal_deaths*62.63

    cost = 0

    low_pph = np.sum(df_aggregate.loc[timepoint, 'Complications-Health'], axis=1)[0]*np.array([1-p_severe, p_severe])[0]
    high_pph = np.sum(df_aggregate.loc[timepoint, 'Complications-Health'], axis=1)[0]*np.array([1-p_severe, p_severe])[1]
    maternal_deaths = np.sum(df_aggregate.loc[timepoint, 'Complications-Health'], axis=0)[0]
    i_DALYs = DALYs[0]*low_pph*62.63 + DALYs[1]*high_pph*62.63 + DALYs[2]*maternal_deaths*62.63
    
    DALYs_averted = b_DALYs - i_DALYs
    if flag_sdr: 
        cost += 2904051
    if flag_int1: 
        cost += 1*(low_pph + high_pph)
    if flag_int2:
        cost += 2.26*0.555*np.sum(df_aggregate.loc[timepoint, 'Live Births Final'])*supply_added
    if flag_int5:
        eclampsia = np.sum(df_aggregate.loc[timepoint, 'Complications-Health'], axis=1)[2]*supply_added
        cost += 3*eclampsia
    if flag_int6:
        sepsis = np.sum(df_aggregate.loc[timepoint, 'Complications-Health'], axis=1)[1]*supply_added
        cost += 12.30*sepsis

    return cost, DALYs_averted

def run_model(subcounty, flags, i_scenarios, sc, global_vars):

    i_INT, Capacity = set_flags(subcounty, flags, i_scenarios, sc, global_vars)

    p_anemia = 0.25
    p_pph = 0.017
    p_sepsis = 0.00159
    p_eclampsia = 0.0046

    i_subcounty=[]
    i_time=[]

    p_anc_anemia = []
    p_anemia_pph = []
    p_anemia_sepsis = []
    p_anemia_eclampsia = []

    anemia_pph = np.array(odds_prob(3.54, p_pph, p_anemia)) * i_INT['int2']['effect']
    anemia_sepsis = np.array(odds_prob(5.68, p_sepsis, p_anemia)) * i_INT['int2']['effect']
    anemia_eclampsia = np.array(odds_prob(3.74, p_eclampsia, p_anemia)) * i_INT['int2']['effect']

    for j in range(sc_time['n']):
        if flags[2]!=1:
            anc_anemia = np.array(odds_prob(2.26, p_anemia, 1-sc['ANC'][j])) * (1-sc['ANC'][j]*i_INT['ANC']['effect'][j,0])/(1-sc['ANC'][j])
            for i in range(n_months):
                p_anc_anemia.append(anc_anemia)
                i_subcounty.append(j)
                i_time.append(i)
                p_anemia_pph.append(anemia_pph)
                p_anemia_sepsis.append(anemia_sepsis)
                p_anemia_eclampsia.append(anemia_eclampsia)
        else:
            for i in range(n_months):
                anc_anemia = np.array(odds_prob(2.26, p_anemia, 1-sc['ANC'][j])) * (1-sc['ANC'][j]*i_INT['ANC']['effect'][j,0])/(1-sc['ANC'][j])
                p_anc_anemia.append(anc_anemia)
                i_subcounty.append(j)
                i_time.append(i)
                p_anemia_pph.append(anemia_pph)
                p_anemia_sepsis.append(anemia_sepsis)
                p_anemia_eclampsia.append(anemia_eclampsia)
                
    probs = pd.DataFrame({'Subcounty': i_subcounty, 'Time': i_time, 'p_anc_anemia': p_anc_anemia, 'p_anemia_pph': p_anemia_pph, 'p_anemia_sepsis': p_anemia_sepsis, 'p_anemia_eclampsia': p_anemia_eclampsia})
    
    flag_pushback = global_vars[1]
    Push_back = np.zeros((n_months,sc_time['n']))

    index = pd.MultiIndex.from_product([range(sc_time['n']), range(n_months)], names=['Subcounty', 'Time'])

    df_3d = pd.DataFrame(index=index, columns=['Live Births Initial', 'Live Births Final', 'LB-ANC', 'ANC-Anemia', 'Anemia-Complications',
                                            'Complications-Facility Level', 'Facility Level-Complications Pre','Transferred In', 'Transferred Out', 'Facility Level-Complications Post', 'Facility Level-Severity', 'Transfer Severity Mortality', 'Deaths', 'Facility Level-Health', 'Complications-Survive', 'Complications-Health', 'Complications-Complications', 
                                            'Capacity Ratio', 'Push Back'])

    for i in t:
        LB_tot_i = np.zeros(4)
        for j in range(sc_time['n']):

            if i>0:
                sc_time['LB1s'][j][i,:],Push_back[i,j] = f_LB_effect(sc_time['LB1s'][j][i-1,:], i_INT, flag_pushback, j, i, Capacity_ratio[:,j], opinion[i,j])
                LB_tot_i = np.maximum(sc_time['LB1s'][j][0,:], 1)
                SDR_multiplier = sc_time['LB1s'][j][i,:] / LB_tot_i

                anc = sc['ANC'][j] * i_INT['ANC']['effect'][j,i]
                LB_tot, LB_tot_final, lb_anc, anc_anemia, anemia_comps_pre, comps_level_pre, f_comps_level_pre, transferred_in, transferred_out, f_comps_level_post, f_severe_post, f_state_mort, f_deaths, f_health, f_comp_survive, comps_health, comps_comps = f_MM(LB_tot_i, anc, i_INT, SDR_multiplier, f_transfer_rates, probs, j, i)
            
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
                df_3d.loc[(j, i), 'Push Back'] = Push_back[i,j]

            Capacity_ratio[i,j] = np.sum(sc_time['LB1s'][j][i,2:4]) / Capacity[j]
            opinion[i,j] = np.sum(sc_time['LB1s'][j][i,2:4]) / np.sum(sc_time['LB1s'][j][i,:])
            df_3d.loc[(j, i), 'Capacity Ratio'] = Capacity_ratio[i,j]
            LB_tot_i += sc_time['LB1s'][j][i,:]

    return df_3d

def f_LB_effect(SC_LB_previous, INT, flag_pushback, j, i, Capacity_ratio, opinion):
    INT_b = INT['CHV']['sdr_effect'][j,i]
    INT_c = INT['Community']['effect'][j]*community_effect(opinion)*(INT_b>0)
    if flag_pushback:
        i_months = range(max(0, i-INT['CHV']['n_months_push_back']), i)
        mean_capacity_ratio = np.mean(Capacity_ratio[i_months])
        push_back = max(0, mean_capacity_ratio-1)
        scale = np.exp(-push_back*INT['CHV']['b_pushback'])
        INT_b = INT_b*scale + INT_c
        effect = np.array([np.exp(-INT_b * np.array([1, 1])), 1 + INT_b * np.array([1, 1])]).reshape(1,4)  # effect of intervention on LB
        SC_LB = SC_LB_previous * effect  # new numbers of LB
        SC_LB = SC_LB / np.sum(SC_LB) * np.sum(SC_LB_previous)  # sum should equal the prior sum
        return SC_LB, push_back
    else:
        INT_b = INT_b + INT_c
        effect = np.array([np.exp(-INT_b * np.array([1, 1])), 1 + INT_b * np.array([1, 1])]).reshape(1,4)  # effect of intervention on LB
        SC_LB = SC_LB_previous * effect  # new numbers of LB
        SC_LB = SC_LB / np.sum(SC_LB) * np.sum(SC_LB_previous)  # sum should equal the prior sum
        return SC_LB, 0

def f_MM(LB_tot, anc, INT, SDR_multiplier_0, f_transfer_rates, probs, j, i):
    LB_tot = np.array(LB_tot) * SDR_multiplier_0   

    p_anc_anemia = probs.loc[(probs['Subcounty']==j) & (probs['Time']==i), 'p_anc_anemia'].item()
    p_anemia_pph = probs.loc[(probs['Subcounty']==j) & (probs['Time']==i), 'p_anemia_pph'].item()
    p_anemia_sepsis = probs.loc[(probs['Subcounty']==j) & (probs['Time']==i), 'p_anemia_sepsis'].item()
    p_anemia_eclampsia = probs.loc[(probs['Subcounty']==j) & (probs['Time']==i), 'p_anemia_eclampsia'].item()

    original_comps = LB_tot * comps_overall
    new_comps = original_comps - LB_tot * f_refer * INT['refer']['effect'][j,i]
    f_comps_adj = new_comps/(LB_tot + new_comps - original_comps)

    lb_anc = np.array([1-anc,anc]) * np.sum(LB_tot) ### lb_anc
    anc_anemia = np.array([         
        [lb_anc[0] * (1-p_anc_anemia[0]), lb_anc[1] * (1-p_anc_anemia[1])],
        [p_anc_anemia[0] * lb_anc[0], p_anc_anemia[1] * lb_anc[1]]
    ])  ### anc_anemia

    anemia_comps = np.array([
    [p_anemia_pph[1],p_anemia_pph[0]],
    [p_anemia_sepsis[1],p_anemia_sepsis[0]],
    [p_anemia_eclampsia[1], p_anemia_eclampsia[0]],
    [0.5*p_ol, 0.5*p_ol],
    [0.5*p_other, 0.5*p_other]
    ])
    
    anemia_comps_pre = anemia_comps * np.sum(anc_anemia, axis=1) ### complications by no anemia, anemia - anemia_comp
    f_comps_prop = (LB_tot * f_comps_adj) / np.sum(LB_tot * f_comps_adj)
    comp_reduction = np.ones((5,4))
    comp_reduction[0,2:4] = INT['int1']['effect'][j,i] # PPH
    comp_reduction[2,2:4] = INT['int5']['effect'][j,i] # eclampsia
    comps_level_pre = np.sum(anemia_comps_pre,axis=1)[:, np.newaxis] * f_comps_prop # complications by facility level
    comps_level_pre = comps_level_pre * comp_reduction
    anemia_comp_props = anemia_comps_pre/(np.sum(anemia_comps_pre, axis=1)[:, np.newaxis])
    change = (np.sum(anemia_comps_pre, axis=1) - np.sum(comps_level_pre, axis=1))[:,np.newaxis] * anemia_comp_props
    anemia_comps_pre = anemia_comps_pre - change
    #anemia_comps_pre

    f_comps_level_pre = np.sum(comps_level_pre, axis=0)
    f_severe_pre = f_comps_level_pre[:, np.newaxis] * np.array([1-p_severe, p_severe])
    transfer = (1-sc['CLASS'][j]) + INT['transfer']['effect'][j,i]*sc['CLASS'][j]
    f_transfer_rates = f_transfer_rates * transfer
    transferred_in = f_severe_pre[:,1] @ f_transfer_rates
    transferred_out = np.sum(f_severe_pre[:,1].reshape(1,4) * f_transfer_rates.T, axis=0)
    f_comps_level_post = f_comps_level_pre + transferred_in - transferred_out

    f_severe_post = f_comps_level_pre[:, np.newaxis] * np.array([1-p_severe, p_severe]) + np.array([np.zeros(4),transferred_in]).T - np.array([np.zeros(4),transferred_out]).T
    # ## in case where more transfers than severe complications   
    # severe, not transferred, severe transferred, not severe 
    f_state_mort = np.c_[f_comps_level_pre[:, np.newaxis] * np.array([1-p_severe, p_severe]) - np.array([np.zeros(4),transferred_out]).T, transferred_in]
    f_deaths = np.sum(f_state_mort * f_mort_rates, axis=1)
    f_comp_survive = f_state_mort - f_state_mort * f_mort_rates
    f_health = np.c_[f_deaths, np.sum(f_comp_survive, axis=1)]

    temp = sum(np.sum(comps_level_pre, axis=1) * b_comp_mort_rates) 
    comp_props = np.sum(comps_level_pre, axis=1) * b_comp_mort_rates/temp
    comps_health = np.c_[np.sum(f_deaths)*comp_props, np.sum(comps_level_pre, axis=1)-np.sum(f_deaths)*comp_props] ### 
    mort_reduction = np.zeros(5)
    mort_reduction[1] = INT['int6']['effect'][j, i]
    mort_reduction = mort_reduction * comps_health[:, 0]  # sepsis
    comps_health[:,0] = comps_health[:,0] - mort_reduction
    comps_health[:,1] = comps_health[:,1] + mort_reduction

    comps_comps = f_severe_pre[:,1].reshape(4,1) * f_transfer_rates
    for i in range(4):
        comps_comps[i,i] = f_comps_level_pre[i] - np.sum(comps_comps,axis=1)[i]

    LB_tot_final = LB_tot + transferred_in - transferred_out

    return LB_tot, LB_tot_final, lb_anc, anc_anemia, anemia_comps_pre, comps_level_pre, f_comps_level_pre, transferred_in, transferred_out, f_comps_level_post, f_severe_post, f_state_mort, f_deaths, f_health, f_comp_survive, comps_health, comps_comps