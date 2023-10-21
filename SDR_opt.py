import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import pandas as pd
import streamlit as st
# OPTIMIZATION
# figure out where the mother goes for ANC initially
def opt_anc(facilities, mothers, df_data, cost_matrix):

    A = np.zeros((len(facilities) + len(facilities) + len(facilities) * 3, len(facilities) * len(facilities) * 3))
    sparseMatrix = sp.lil_matrix((A.shape),
                                 dtype=np.int8)
    looks_test = {}
    value = 0
    for x in range(len(facilities)):
        for y in range(len(facilities)):
            for k in range(3):
                looks_test[x, y, k] = value
                value += 1

    # capacity at the destination
    row = 0
    for j in range(len(facilities)):
        for i in range(len(facilities)):
            for z in range(3):
                column = looks_test[i, j, z]
                sparseMatrix[row, column] = 1
        row += 1

    # capacity at the starting location
    for i in range(len(facilities)):
        for j in range(len(facilities)):
            for z in range(3):
                column = looks_test[i, j, z]
                sparseMatrix[row, column] = 1
        row += 1

    # risk capacity at the starting location
    for i in range(len(facilities)):
        for z in range(3):
            for j in range(len(facilities)):
                column = looks_test[i, j, z]
                sparseMatrix[row, column] = 1
            row += 1

    # DELIVERY CONSTRAINT
    capacity = list(df_data['live_birth'])
    capacity.append(50000)

    # STARTING CONSTRAINT
    starting = list(df_data['anc_new'])
    starting.append(0)

    ### GET RISK CONSTRAINT
    facility_anc = [mother.facility_anc.idx for mother in mothers]
    facility_risk = [mother.risk for mother in mothers]
    anc_group = pd.DataFrame({'ANC': facility_anc, 'Risk': facility_risk})
    key = df_data['Key']
    key.loc[399] = 430
    ANC = anc_group.groupby('ANC').value_counts().reset_index()
    risk_cap = []
    for i in key:
        for j in range(3):
            if i in list(ANC['ANC']) and (j in list(ANC['Risk'])):
                if len(ANC.loc[(ANC['ANC'] == i) & (ANC['Risk'] == j),]) > 0:
                    risk_cap.append(ANC.loc[(ANC['ANC'] == i) & (ANC['Risk'] == j), 0].values[0])
                else:
                    risk_cap.append(0)
            else:
                risk_cap.append(0)
    risk_cap[1197:] = [5000, 5000, 5000]

    # OPTIMIZATION
    b = capacity + starting + risk_cap

    m = gp.Model("matrix1")
    x = m.addMVar(shape=A.shape[1], vtype=GRB.INTEGER, name="x")
    obj = cost_matrix
    m.setObjective(obj @ x, GRB.MINIMIZE)
    A = sparseMatrix.tocsr()
    rhs = np.array(b)

    # number at each end facility must be less than or equal to the capacity at the facility
    m.addConstr(A[0:len(facilities), :] @ x <= rhs[0:len(facilities)])
    # # number at each starting facility must be equal to the original capacity at the facility
    m.addConstr(A[len(facilities):(len(facilities) + len(facilities)), :] @
                x == rhs[len(facilities):(len(facilities) + len(facilities))])
    # # number at each starting facility by risk must be equal to the original at the facility
    m.addConstr(
        A[(len(facilities) + len(facilities)):(len(facilities) + len(facilities) + len(facilities) * 3 - 3), :] @
        x <= rhs[(len(facilities) + len(facilities)):(len(facilities) + len(facilities) + len(facilities) * 3 - 3)])
    m.optimize()

    X = x.X

    X = X.reshape((len(facilities), len(facilities), 3))
    key_solution = {}
    for i in range(len(facilities)):
        for j in range(len(facilities)):
            for z in range(3):
                key_solution[i, j, z] = X[i, j, z]

    return key_solution
