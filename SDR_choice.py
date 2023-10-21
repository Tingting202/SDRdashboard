import numpy as np
import pandas as pd


# FOR CHOICE
# FOR CHOICE
def get_choices(facility_object, l1, l2, df, distance, ref_count, df_births):
    """From a facility with a complication, get dataframe of choice characteristics"""
    facility = facility_object.idx
    # origin_cat = df['Categorization'][df['Key'] == facility]
    # complication = complication # check this out
    l1_dist = []
    l2_dist = []
    l1_choice = {}
    l2_choice = {}
    l3_choice = {}
    # get facility of minimal distance from each category, get characteristics
    for i in l1:
        l1_dist.append(distance[df[df['Key'] == facility].index[0], df[df['Key'] == i].index[0]])
        if min(l1_dist) == 0:
            l1_dist[np.argmin(l1_dist)] = 100
        l1_choice['Facility'] = l1[np.argmin(l1_dist)]
        l1_choice['Distance'] = min(l1_dist)
        # l1_choice['Ref_Count'] = ref_count[df[df['Key']==l1_choice['Facility']].index[0],df[df['Key']==i].index[0]]
    l1_choice['Ref_Count'] = ref_count[
        df[df['Key'] == facility].index[0], df[df['Key'] == l1_choice['Facility']].index[0]]
    l1_choice['Births'] = df_births[df['Key'] == l1_choice['Facility']].values[0]
    l1_choice['Category'] = 1
    for i in l2:
        l2_dist.append(distance[df[df['Key'] == facility].index[0], df[df['Key'] == i].index[0]])
        if min(l2_dist) == 0:
            l2_dist[np.argmin(l2_dist)] = 100
        l2_choice['Facility'] = l2[np.argmin(l2_dist)]
        l2_choice['Distance'] = min(l2_dist)
        # l2_choice['Ref_Count'] = ref_count[df[df['Key']==l2_choice['Facility']].index[0],df[df['Key']==i].index[0]]
    l2_choice['Ref_Count'] = ref_count[
        df[df['Key'] == facility].index[0], df[df['Key'] == l2_choice['Facility']].index[0]]
    l2_choice['Births'] = df_births[df['Key'] == l2_choice['Facility']].values[0]
    l2_choice['Category'] = 2
    l3_choice['Facility'] = 151
    l3_choice['Distance'] = distance[df[df['Key'] == facility].index[0], df[df['Key'] == 151].index[0]]
    l3_choice['Ref_Count'] = ref_count[df[df['Key'] == facility].index[0], df[df['Key'] == 151].index[0]]
    l3_choice['Births'] = df_births[df['Key'] == 151].values[0]
    l3_choice['Category'] = 3

    dict_list = [l1_choice, l2_choice, l3_choice]

    categ = df['Level Facility'][df['Key'] == facility].values[0]
    return (pd.DataFrame(dict_list), categ)


def get_probs(choices, categ, complication, coef):
    sums = 0
    sums += coef[0]
    sums += np.multiply(np.matrix(choices['Births']), coef[1])
    if categ == 2:
        sums += coef[2]
    elif categ == 4:
        sums += coef[3]
    else:
        sums += 0
    if complication == 1:
        sums += coef[4]
    elif complication == 2:
        sums += coef[5]
    else:
        sums += 0
    sums += np.multiply(np.matrix(choices['Distance']), coef[6])
    sums += np.multiply(np.matrix(choices['Ref_Count']), coef[7])
    probs = np.exp(sums) / np.sum(np.exp(sums))
    probs = np.array(probs).reshape(-1)

    return (probs)
