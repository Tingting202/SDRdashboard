#!/usr/bin/env python
# coding: utf-8

# In[50]:

import mesa
from mesa import Agent
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
# from mesa.visualization.UserParam import UserSettableParameter

import numpy as np
import pandas as pd
import seaborn as sns
import random
from random import randrange, uniform
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
import scipy.stats as ss
import streamlit as st
import time
import plotly.graph_objects as go


# In[51]:

##Parameters##
st.set_page_config(layout="wide")
st.title("SDR ABM Simulation")

# Define slider parameters
col1, col2, col3, col4 = st.columns(4)
with col1:
    ANC_prefer_mean = st.slider("Risk-differentiated ANC Preference Mean", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
    hist_risk_mean = st.slider("Historical Risk Mean", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
with col2:
    AI_ultrasound_rate = st.slider("AI Ultrasound Rate in Risk-differentiated ANC", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
with col3:
    l45_rate_if_no_know = st.slider("L45 Rate if No Knowledge", min_value=0.0, max_value=1.0, step=0.1, value=0.1)
    l45_rate_if_low_know = st.slider("L45 Rate if Low Knowledge", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
    l45_rate_if_high_know = st.slider("L45 Rate if High Knowledge", min_value=0.0, max_value=1.0, step=0.1, value=0.9)
with col4:
    sim_years = st.slider("Simulation Years", 1, 10, 3, 1)
#Subcounty names
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

#simulation years
# sim_years = 3

#pregnant mothers by subcounty in one year
sub_pops = np.array([5944, 4143, 3890, 4988, 6738, 6700, 8508, 6158, 4079, 4609, 5730, 6274]) * sim_years

#Probability of selecting initial delivery location
init_loc_prop = [
    [0.32, 0.28, 0.38, 0.02],
    [0.43, 0.16, 0.38, 0.02],
    [0.44, 0.21, 0.33, 0.02],
    [0.17, 0.40, 0.41, 0.02],
    [0.44, 0.21, 0.33, 0.02],
    [0.20, 0.21, 0.01, 0.58],
    [0.45, 0.14, 0.39, 0.02],
    [0.38, 0.38, 0.22, 0.02],
    [0.43, 0.34, 0.21, 0.02],
    [0.12, 0.37, 0.49, 0.02],
    [0.38, 0.32, 0.28, 0.02],
    [0.40, 0.29, 0.30, 0.02]
]

#probability of having ANC at baseline
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

def get_prob_comps():
    param = {
        'p_comps_anc':0.0596/2,             #Probability of having complications if ANC
        'p_comps_noanc':0.0861/2,           #Probability of having complications if no ANC
        'p_type_comps': [0.1, 0.04, 0.12, 0.27, 0.01, 0.45], #Probability of having each type of complication (aph, sepsis, eclampsia, obstructed labor, ruptured uterus, pph) if with complications 
        'p_deliver_loc_comps': np.array([
            [0.2035,0.1761,0.6198],
            [0.3059,0.1051,0.5930],
            [0.2524,0.1325,0.6280],
            [0.1120,0.2450,0.6575],
            [0.2255,0.1274,0.6500],
            [0.3316,0.1394,0.5233],
            [0.2901,0.0878,0.6167],
            [0.1430,0.2403,0.6260],
            [0.1926,0.2152,0.5882],
            [0.1947,0.2406,0.5524],
            [0.1564,0.1979,0.6520],
            [0.1996,0.1797,0.6193]
        ]), #Probability of delivering at home, L23, L45 if with complications by subcounty
        'p_deliver_loc_nocomps': np.array([
            [0.3301,0.2857,0.3839],
            [0.4797,0.1648,0.3551],
            [0.4040,0.2121,0.3837],
            [0.1841,0.4027,0.4126],
            [0.3750,0.2119,0.4127],
            [0.4942,0.2078,0.2977],
            [0.4732,0.1432,0.3840],
            [0.2297,0.3858,0.3838],
            [0.3046,0.3403,0.3552],
            [0.3014,0.3725,0.3265],
            [0.2593,0.3282,0.4127],
            [0.3241,0.2917,0.3839]
         ]), #Probability of delivering at home, L23, L45 if no complications by subcounty
        'p_l5_l45': 0.13,                   #Probability of delivering at L5 if delivering at L45
        'p_comp_severe': 0.216,             #Probability of having severe complications
        'm_l_home': 0.005 / 100,            #Probability of maternal death if delivering at home and with low complications
        'm_l_l23': 0.004 / 100,             #Probability of maternal death if delivering at L23 and with low complications
        'm_l_l4': 0.004 / 100,              #Probability of maternal death if delivering at L4 and with low complications
        'm_l_l5': 0.003 / 100,              #Probability of maternal death if delivering at L5 and with low complications
        'm_h_home': 25 / 100,               #Probability of maternal death if delivering at home and with high complications
        'm_h_l23': 5.18 / 100,              #Probability of maternal death if delivering at L23 and with high complications
        'm_h_l4': 5 / 100,                  #Probability of maternal death if delivering at L4 and with high complications
        'm_h_l5': 4.93 / 100,               #Probability of maternal death if delivering at L5 and with high complications
        'p_transfer_l4_home': 24.7/100,     #Probability of transferring from home to L4
        'p_transfer_l5_home': 10.335/100,   #Probability of transferring from home to L5
        'p_transfer_l4_l23': 49.36/100,     #Probability of transferring from L23 to L4
        'p_transfer_l5_l23': 24.422/100,    #Probability of transferring from L23 to L5
        'p_transfer_l5_l4': 21.684/100,     #Probability of transferring from L4 to L5
        'm_transfer': 3.35,                 #Multiplier for maternal death if emergency transfer was made
        'm_c_section': 0.5,                  #Multiplier for maternal death if c-section was made
        'l45_capacity': np.array([3000.0, 1992.0, 1608.0, 2328.0, 2784.0, 6060.0, 4176.0, 2736.0, 1308.0, 5820.0, 2088.0, 5124.0]) / 12  #Capacity of L45 facilities by subcounty
}
    return param

param= get_prob_comps()


# In[52]:


##Agents and Model
class Mother(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        ##Demographics##
        self.subcounty_id = None                        #Subcounty ID
        self.subcounty = None                           #Subcounty agent    
        self.pregnancy_time = randrange(0, self.model.ticks)       #Time of pregnancy
        self._pregnant = False                          #Whether pregnant
        self.base_ANC_preference = np.random.normal(ANC_prefer_mean, 0.1, size=1) #ANC preference at baseline level
        self.historical_risk = np.random.normal(hist_risk_mean, 0.1, size=1)     #Historical risk of complications
        self.knowledge_risk = 'None'                    #The status of knowledge of risk
        
        ##Maternal Health Services##
        self.prob_ANC = None                            #Probability of having ANC
        self._ANC = False                               #Whether has ANC services
        self._ANCtype = 'None'                          #Type of ANC services, basic ANC or risk-differentiated ANC
        self.prob_AI_ultrasound = None                  #Probability of having AI ultrasound
        self._AI_ultrasound = False                     #Whether has AI ultrasound services
        
        ##Complications and death##
        self.prob_comp_risk = None                      #Probability of developing complication risks
        self._comp_risk = False                         #Whether has complication risks
        self._comps = False                             #Whether has complications
        self._ante_comps = False                        #Whether has antepartum complications
        self._intra_comps = False                       #Whether has intrapartum complications
        self._post_comps = False                        #Whether has postpartum complications
        self._PPH = False                               #Whether has pph
        self._sepsis = False                            #Whether has sepsis
        self._eclampsia = False                         #Whether has eclampsia
        self._obstructed = False                        #Whether has obstructed labor
        self._aph = False                               #Whether has Antepartum Hemorrhage
        self._ruptured = False                          #Whether has Ruptured Uterus
        self._comp_severe = False                       #Whether the complication is severe or not
        self._death = False                             #Whether the mother died
        
        ##Delivery##       
        self.init_deliver_level = None                  #Initial delivery location level
        self._emergenct_transfer = False                #Whether emergency transfer was made
        self.final_deliver_level = None                 #Final delivery location level
        self.final_deliver_loc = None                   #Final delivery location
        self.delivery_method = None                     #Delivery method, c-section or vaginal delivery
        self._exit = False                              #Time to finish all maternal care
        self._delivered = False                         #Whether the baby was delivered
    
    def whether_ANC(self):
        if random.random() <= self.prob_ANC:
            self._ANC = True
            if (self.base_ANC_preference >= 0.5) or (self.historical_risk >= 0.5):
                self.ANCtype = 'Risk-differentiated'
                self.prob_AI_ultrasound = AI_ultrasound_rate
            else:
                self.ANCtype = 'Basic'
                self.prob_AI_ultrasound = 0.2
        else:
            self._ANC = False
            self.ANCtype = 'None'
            self.prob_AI_ultrasound = 0
            
    def AI_ultrasound(self):
        if random.random() <= self.prob_AI_ultrasound:
            self._AI_ultrasound = True
    
    def develop_complication_risks(self):
        ANC_factor = 0
        if self.ANCtype == 'Basic':
            ANC_factor = 0.1
        if self.ANCtype == 'Risk-differentiated':
            ANC_factor = 0.2
        
        self.prob_comp_risk = max(self.historical_risk - ANC_factor, 0)
        if random.random() <= self.prob_comp_risk:
            self._comp_risk = True
    
    def update_knowledge_of_risk(self):
        if self._comp_risk:
            if self._AI_ultrasound:
                self.knowledge_risk = 'High'
            elif self._ANC:
                self.knowledge_risk = 'Low'
            else:
                self.knowledge_risk = 'None'
        else:
            self.knowledge_risk = 'None'

    def select_init_deliver_loc(self):
        facility_probabilities = [
            [(1-l45_rate_if_no_know)*0.5, (1-l45_rate_if_no_know)*0.5, l45_rate_if_no_know], #None knowledge
            [(1-l45_rate_if_low_know)*0.3, (1-l45_rate_if_low_know)*0.7, l45_rate_if_low_know], #Low knowledge
            [(1-l45_rate_if_high_know)*0.1, (1-l45_rate_if_high_know)*0.9, l45_rate_if_high_know]  #High knowledge
        ]
        facility_choices = ['home', 'l23', 'l45']
        
        if self.knowledge_risk == 'None': 
            deliver_loc = random.choices(facility_choices, weights=facility_probabilities[0])[0]
        elif self.knowledge_risk == 'Low':
            deliver_loc = random.choices(facility_choices, weights=facility_probabilities[1])[0]
        elif self.knowledge_risk == 'High':
            deliver_loc = random.choices(facility_choices, weights=facility_probabilities[2])[0]
            
        if deliver_loc == 'home':
            self.init_deliver_level = 0
        elif deliver_loc == 'l23':
            self.init_deliver_level = 1
        elif deliver_loc == 'l45':
            if self.subcounty_id !=5:
                if random.random() <= param['p_l5_l45']:
                    self.init_deliver_level = 3
                else:
                    self.init_deliver_level = 2
            else:
                self.init_deliver_level = 3    
    
    def emergency_transfer(self):
        self.final_deliver_level = self.init_deliver_level
        if self._comp_severe:
            if self.init_deliver_level == 0:
                if random.random() <= param['p_transfer_l4_home']:
                    self.final_deliver_level = 2
                    self._emergenct_transfer = True
                elif random.random() <= param['p_transfer_l5_home']:
                    self.final_deliver_level = 3
                    self._emergenct_transfer = True
            elif self.init_deliver_level == 1:
                if random.random() <= param['p_transfer_l4_l23']:
                    self.final_deliver_level = 2
                    self._emergenct_transfer = True
                elif random.random() <= param['p_transfer_l5_l23']:
                    self.final_deliver_level = 3
                    self._emergenct_transfer = True
            elif self.init_deliver_level == 2:
                if random.random() <= param['p_transfer_l5_l4']:
                    self.final_deliver_level = 3
                    self._emergenct_transfer = True
        
        for facility in self.model.facilities:
            if (facility.level == self.final_deliver_level) and (facility.subcounty_id == self.subcounty_id):
                self.final_deliver_loc = facility
                self.final_deliver_loc.queue.append(self)
    
    def delivery_methods(self):
        if self.final_deliver_level == 0:
            self.delivery_method = 'Vaginal'
        elif self.final_deliver_level == 1:
            self.delivery_method = 'Vaginal'
        elif self.final_deliver_level == 2 or self.final_deliver_level == 3:
            if self._comp_risk or self._comps or self._comp_severe:
                self.delivery_method = 'C-section'
            else:
                self.delivery_method = 'Vaginal'
    
    def develop_complications(self):
        if self.delivery_method == 'C-section':
            c_section_factor_obstructed = 0.01
            c_section_factor_ph = 0.15
            c_section_factor_other = 0.5
        else:
            c_section_factor_obstructed = 1
            c_section_factor_ph = 1
            c_section_factor_other = 1
            
        if self._comp_risk and random.random() <= 0.5:
            comps_choices = ['APH', 'Sepsis', 'Eclampsia', 'Obstructed', 'Ruptured', 'PPH']
            comps_type_probs = param['p_type_comps']
            
            type = random.choices(comps_choices, weights=comps_type_probs)[0]
            if type == 'APH':
                if random.random() <= c_section_factor_ph:
                    self._aph = True
                    self._ante_comps = True
            elif type == 'Sepsis':
                self._sepsis = True
                self._intra_comps = True
            elif type == 'Obstructed':
                if random.random() <= c_section_factor_obstructed:
                    self._obstructed = True
                    self._intra_comps = True
            elif type == 'Ruptured':
                if random.random() <= c_section_factor_other:
                    self._ruptured = True
                    self._intra_comps = True
            elif type == 'Eclampsia':
                if random.random() <= c_section_factor_other:
                    self._eclampsia = True
                    self._intra_comps = True
            elif type == 'PPH':
                if random.random() <= c_section_factor_ph:
                    self._PPH = True
                    self._post_comps = True
                    
            if self._ante_comps or self._intra_comps or self._post_comps:
                if random.random() <= param['p_comp_severe']:
                    self._comp_severe = True
                    self._comps = True
    
    def deliver_baby(self):
        self._delivered = True
        
    def develop_MNH_outcomes(self):
        if self._emergenct_transfer:
            emergency_multiplier = param['m_transfer']
        else:
            emergency_multiplier = 1
        
        if self.delivery_method == 'C-section' and (self._aph or self._eclampsia or self._ruptured or self._PPH or self._obstructed):
            c_section_reduction = param['m_c_section']
        else:
            c_section_reduction = 1
        
        if self._comps:
            if self._comp_severe:
                if self.final_deliver_level == 0:
                    if random.random() <= param['m_h_home'] * emergency_multiplier:
                        self._death = True
                elif self.final_deliver_level == 1:
                    if random.random() <= param['m_h_l23'] * emergency_multiplier:
                        self._death = True
                elif self.final_deliver_level == 2:
                    if random.random() <= param['m_h_l4'] * emergency_multiplier * c_section_reduction:
                        self._death = True
                elif self.final_deliver_level == 3:
                    if random.random() <= param['m_h_l5'] * emergency_multiplier * c_section_reduction:
                        self._death = True
            else:
                if self.final_deliver_level == 0:
                    if random.random() <= param['m_l_home'] * emergency_multiplier:
                        self._death = True
                elif self.final_deliver_level == 1:
                    if random.random() <= param['m_l_l23'] * emergency_multiplier:
                        self._death = True
                elif self.final_deliver_level == 2:
                    if random.random() <= param['m_l_l4'] * emergency_multiplier * c_section_reduction:
                        self._death = True
                elif self.final_deliver_level == 3:
                    if random.random() <= param['m_l_l5'] * emergency_multiplier * c_section_reduction:
                        self._death = True            
        
    def record_data_to_fac_tables(self):
        self.model.fac_tables.loc[(self.subcounty_id, self.final_deliver_level, self.model._current_tick), ['Live Births']] += 1
        if self._comps:
            self.model.fac_tables.loc[(self.subcounty_id, self.final_deliver_level, self.model._current_tick), ['Complications']] += 1
        if self._death:
            self.model.fac_tables.loc[(self.subcounty_id, self.final_deliver_level, self.model._current_tick), ['Deaths']] += 1
    
    def discharge(self):
        self._ANC = False
        self._pregnant = False
        self.final_deliver_loc.queue.remove(self)
        self._exit = True
    
    def step(self):
        #Whether pregnant, if pregnant
        if (self._pregnant == False) & (self.model._current_tick == self.pregnancy_time):
            self._pregnant = True
            self.whether_ANC()
            self.AI_ultrasound()
        
        if self._pregnant == True & (self.model._current_tick == self.pregnancy_time + 4):
            self.develop_complication_risks()
            self.update_knowledge_of_risk()

        if (self._pregnant == True) & (self.model._current_tick == self.pregnancy_time + 9):   
            self.select_init_deliver_loc()
            self.delivery_methods()
            self.develop_complications()
            self.emergency_transfer()
            self.delivery_methods()
        
        if (self._pregnant == True) & (self.model._current_tick == self.pregnancy_time + 10):
            self.deliver_baby()
            self.develop_MNH_outcomes()
            #self.record_data_to_fac_tables()
        
        #Dischage and exit the model
        if (self.model._current_tick == (self.pregnancy_time + 11)):
            self.discharge()
                         
class Facility(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.queue = []                                 #Restore mothers who deliver at this facility level                    
        self.subcounty_id = None                        #Subcounty ID
        self.subcounty = None                           #Subcounty agent
        self.level = None                               #Level of facility
        self.capacity = None                            #Facility capacity

class Subcounty(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.subcounty_id = None
        self.mothers = []

class SDRModel(Model):
    def __init__(self, ticks):
        super().__init__()
        self.ticks = ticks
        self._current_tick = 0
        self.agents_to_remove = []
        self.schedule = RandomActivation(self)
        self.create_subcounties()
        self.create_mothers()
        self.create_facilities()
        self.match_agents()
        self.datacollector = DataCollector(
            model_reporters = {
                'Active Mothers': get_mothers_active,
                'Live births this month': get_live_births_this_month,
                'Mothers Pregnant': get_mothers_pregnant,
                'Live births at Home': get_live_births_at_home,
                'Live births at L23': get_live_births_at_L23,
                'Live births at L4': get_live_births_at_L4,
                'Live births at L5': get_live_births_at_L5,
                'Complication rate': get_complications,
                'Death rate': get_deaths
            }
        )
        self.create_fac_df()
        self.create_sc_df()
        self.running = True
        
    
    def create_subcounties(self):
        self.subcounties = []
        idx = 0
        for i in range(12):
            subcounty = Subcounty(idx, self)
            subcounty.subcounty_id = i
            self.schedule.add(subcounty)
            self.subcounties.append(subcounty)
            idx += 1
        return self.subcounties

    def create_mothers(self):
        self.mothers = []
        idx = len(self.subcounties)
        for i in range(12):
            for j in range(sub_pops[i]):
                mother = Mother(idx, self)
                mother.subcounty_id = i
                mother.prob_ANC = sc_ANC[i]
                self.mothers.append(mother)
                idx += 1
        return self.mothers
    
    def create_facilities(self):
        self.facilities = []
        idx = len(self.mothers) + len(self.subcounties)
        for i in range(12):
            for j in range(4):
                facility = Facility(idx, self)
                facility.subcounty_id = i
                facility.level = j
                if i!= 5 and j == 2:
                    facility.capacity = param['l45_capacity'][i]
                if i == 5 and j == 3:
                    facility.capacity = param['l45_capacity'][i]
                self.schedule.add(facility)
                self.facilities.append(facility)
                idx += 1
        return self.facilities
    
    #To link mothers, facilities, and subcounties
    def match_agents(self):
        for mother in self.mothers:
            for subcounty in self.subcounties: 
                if subcounty.subcounty_id == mother.subcounty_id:
                    mother.subcounty = subcounty
                    subcounty.mothers.append(mother)
        
        for facility in self.facilities:
            for subcounty in self.subcounties: 
                if subcounty.subcounty_id == facility.subcounty_id:
                    facility.subcounty = subcounty
    
    def create_sc_df(self):
        index = pd.MultiIndex.from_product([range(12), range(self.ticks)], names=['Subcounty', 'Time'])
        self.sc_tables = pd.DataFrame(index=index, columns= ['No_Mothers', 'ANC rate', 
                                                          'Live Births-Home', 'Live Births-L23', 
                                                          'Live Births-L4','Live Births-L5'])
        
    def create_fac_df(self):
        index = pd.MultiIndex.from_product([range(12), range(4), range(self.ticks)], names=['Subcounty', 'Facility Level', 'Time'])
        self.fac_tables = pd.DataFrame(index=index, columns= ['Live Births', 'Complications', 'Deaths'])
        self.fac_tables = self.fac_tables.fillna(0)
    
    def generate_df(self):
        for facility in self.facilities:
            no_mothers = len(facility.queue)
             
            if facility.level == 0:
                self.sc_tables.loc[(facility.subcounty_id, self._current_tick),['Live Births-Home']] = no_mothers

            elif facility.level == 1:
                self.sc_tables.loc[(facility.subcounty_id, self._current_tick),['Live Births-L23']] = no_mothers

            elif facility.level == 2:
                self.sc_tables.loc[(facility.subcounty_id, self._current_tick),['Live Births-L4']] = no_mothers

            elif facility.level == 3:
                self.sc_tables.loc[(facility.subcounty_id, self._current_tick),['Live Births-L5']] = no_mothers
        
        for subcounty in self.subcounties:
            no_pregnant_mothers = np.sum([mother._pregnant for mother in subcounty.mothers])
            self.sc_tables.loc[(subcounty.subcounty_id, self._current_tick), ['No_Mothers']] = len(subcounty.mothers)
            
            no_ANC_mothers = np.sum([(mother._pregnant and mother._ANC) for mother in subcounty.mothers])
            self.sc_tables.loc[(subcounty.subcounty_id, self._current_tick), ['ANC rate']] = no_ANC_mothers / no_pregnant_mothers
    
    def step(self):
        for mother in self.mothers:
            if self._current_tick == mother.pregnancy_time:
                self.schedule.add(mother)
        
        self.schedule.step() 
            
        agents_to_remove = [agent for agent in self.schedule.agents if isinstance(agent, Mother) and agent._exit]
        for agent in agents_to_remove:
            self.schedule.remove(agent)
        
        #self.generate_df()
        self._current_tick += 1
        self.datacollector.collect(self)
         

# In[53]:


##Functions to get system outcomes
def get_live_births_this_month(model):
    mothers_delivered = [model._current_tick == (mother.pregnancy_time + 10) for mother in model.mothers]
    no_mothers_delivered = np.sum(mothers_delivered)
    return no_mothers_delivered

def get_mothers_active(model):
    no_mothers_active = sum(1 for agent in model.schedule.agents if isinstance(agent, Mother))
    return no_mothers_active

def get_mothers_pregnant(model):
    mothers_pregnant = [mother._pregnant for mother in model.mothers]
    no_mothers_pregnant = np.sum(mothers_pregnant)
    return no_mothers_pregnant

def get_live_births_at_home(model):
    live_births_home = np.sum([len(facility.queue) for facility in model.facilities if facility.level == 0])
    live_births_all = np.sum([len(facility.queue) for facility in model.facilities])
    lbrate_home = live_births_home / live_births_all
    return lbrate_home

def get_live_births_at_L23(model):
    live_births_23 = np.sum([len(facility.queue) for facility in model.facilities if facility.level == 1])
    live_births_all = np.sum([len(facility.queue) for facility in model.facilities])
    lbrate_23 = live_births_23 / live_births_all
    return lbrate_23

def get_live_births_at_L4(model):
    live_births_4 = np.sum([len(facility.queue) for facility in model.facilities if facility.level == 2])
    live_births_all = np.sum([len(facility.queue) for facility in model.facilities])
    lbrate_4 = live_births_4 / live_births_all
    return lbrate_4

def get_live_births_at_L5(model):
    live_births_5 = np.sum([len(facility.queue) for facility in model.facilities if facility.level == 3])
    live_births_all = np.sum([len(facility.queue) for facility in model.facilities])
    lbrate_5 = live_births_5 / live_births_all
    return lbrate_5

def get_complications(model):
    complications = np.sum([mother._comps for mother in model.mothers])
    mothers_delivered = np.sum([mother._delivered for mother in model.mothers])
    rate = complications / mothers_delivered
    return rate

def get_deaths(model):
    deaths = np.sum([mother._death for mother in model.mothers])
    mothers_delivered = np.sum([mother._delivered for mother in model.mothers])
    rate = deaths / mothers_delivered
    return rate


# In[54]:
# defining containers for the plot
st.markdown('---')

col1, col2, col3 = st.columns(3)
with col1:
    plot1_spot = st.empty()
with col2:
    plot2_spot = st.empty()
with col3:
    plot3_spot = st.empty()

# Run the simulation
def run_simulation():
    ticks = sim_years * 12
    model = SDRModel(ticks=ticks)

    # Loop to call the plot function
    for i in range(ticks):
        model.step()
        run_stats = model.datacollector.get_model_vars_dataframe()

        with plot1_spot:
            livebirth_chart(run_stats)
        with plot2_spot:
            complication_chart(run_stats)
        with plot3_spot:
            death_chart(run_stats)

# function to make chart
def livebirth_chart(run_stats):
    # Plot your data using Plotly or any other library
    fig = go.Figure()
    # Add traces to the figure

    fig.add_trace(
        go.Scatter(x=run_stats.index, y=run_stats['Live births at Home'], mode='lines+markers', name='Live births at Home'))
    fig.add_trace(
        go.Scatter(x=run_stats.index, y=run_stats['Live births at L23'], mode='lines+markers', name='Live births at L23'))
    fig.add_trace(
        go.Scatter(x=run_stats.index, y=run_stats['Live births at L4'], mode='lines+markers', name='Live births at L4'))
    fig.add_trace(
        go.Scatter(x=run_stats.index, y=run_stats['Live births at L5'], mode='lines+markers', name='Live births at L5'))
    # Add more traces as needed
    # Update layout if necessary
    fig.update_layout(title='Live births by facility level', xaxis_title='Month', yaxis_title='# Live Births', \
                      width=500, height=400, legend=dict(orientation="h", yanchor="bottom", y=-0.5)
                      )
    # Show the figure
    st.plotly_chart(fig)

def complication_chart(run_stats):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=run_stats.index, y=run_stats['Complication rate'], mode='lines+markers', name='Complication rate'))
    fig.update_layout(title='Complication rate', xaxis_title='Month', yaxis_title='Complication rate', \
                      width=500, height=400 )

    st.plotly_chart(fig)

def death_chart(run_stats):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=run_stats.index, y=run_stats['Death rate'], mode='lines+markers', name='Death rate'))
    fig.update_layout(title='Death rate', xaxis_title='Month', yaxis_title='Death rate', \
                      width=500, height=400 )

    st.plotly_chart(fig)

if st.button("Run Simulation"):
    run_simulation()


# ticks = 12 * sim_years        #Months to simulate
# model = SDRModel(ticks = ticks)
# for i in range(ticks):
#     model.step()


# TESTING THE CODE------------------------

# In[55]:


# run_stats = model.datacollector.get_model_vars_dataframe()
# run_stats


# In[56]:


# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
# fig.figure.set_figwidth(6)
# fig.figure.set_figheight(10)
# ax1.plot(run_stats[['Active Mothers']])
# ax1.legend(['Mothers in Model Schedule'])
# ax2.plot(run_stats[['Mothers Pregnant']])
# ax2.legend(['Mothers Pregnant'])
# ax3.plot(run_stats[['Live births this month', 'Live births at Home', 'Live births at L23', 'Live births at L4', 'Live births at L5']])
# ax3.legend(['Live births this month', 'Live births at Home', 'Live births at L23', 'Live births at L4', 'Live births at L5'])
# ax4.plot(run_stats[['Complication rate']] * 100)
# ax4.legend(['Complications per 100 live births'])
# ax5.plot(run_stats[['Death rate']] * 1000)
# ax5.legend(['Deaths per 1000 live births'])
# fig.show()


# Old code for plotting below

# In[57]:


# df_3d = model.sc_tables
# df_3d.reset_index(inplace=True)
# df_3d


# In[58]:


# df_4d = model.fac_tables
# df_4d
# test = df_4d.groupby(['Facility Level']).sum()
# test['Complication rate'] = test['Complications'] / test['Live Births']
# test['Death rate'] = test['Deaths'] / test['Live Births']
# test


# In[59]:


# test['Live Births'] / np.sum(test['Live Births'])


# In[60]:


# test = df_4d.sum()
# test['Complication rate'] = test['Complications'] / test['Live Births']
# test['Death rate'] = test['Deaths'] / test['Live Births']
# test


# In[61]:


# fig, axes = plt.subplots(4, 3, figsize=(12, 9))
# 
# for i in range(12):
#     ax = axes[i // 3, i % 3]
#     
#     lines = []
#     for column in ['Live Births-Home', 'Live Births-L23', 'Live Births-L4', 'Live Births-L5']:
#         line, = ax.plot(range(ticks), df_3d.loc[df_3d['Subcounty'] == i, column], label=column)
#         lines.append(line)
#     
#     ax.set_title(f"Subcounty {i}")
#     ax.set_xlabel('Months')
#     ax.set_ylabel('Live Births')
#     
# lines, labels = axes[-1, -1].get_legend_handles_labels()
# fig.legend(lines, labels, loc='lower center', ncol=4)
# 
# plt.tight_layout()
# plt.show()


# In[62]:


# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
# fig.figure.set_figwidth(6)
# fig.figure.set_figheight(10)
# ax1.plot(run_stats[['Active Mothers']])
# ax1.legend(['Mothers in Model Schedule'])
# ax2.plot(run_stats[['Mothers Pregnant']])
# ax2.legend(['Mothers Pregnant'])
# ax3.plot(run_stats[['Live births this month', 'Live births at Home', 'Live births at L23', 'Live births at L4', 'Live births at L5']])
# ax3.legend(['Live births this month', 'Live births at Home', 'Live births at L23', 'Live births at L4', 'Live births at L5'])
# 
# for i in range(12):
#     ax4.plot(range(ticks), df_3d.loc[df_3d['Subcounty'] == i, 'ANC rate'])
# ax4.set_xlabel('Months')
# ax4.set_ylabel('ANC rate')
# 
# fig.show()

