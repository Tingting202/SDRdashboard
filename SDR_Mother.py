import numpy as np


# MOTHER OBJECTS
class Mother:
    def __init__(self, facility_anc, risk):

        self.facility_anc = facility_anc
        self.risk = risk  # this is actually risk
        self.facility_delivery = None

        self.complication = None  # 0, 1 (no complication, yes complication)
        self.complication_level = None  # 0, 1, 2 (no, low, high)
        self.complication_risk = np.array([0, 0.02, 0.07])

        self.final_deliver = None

        self.transfer = None

    def develop_complication(self):
        p = self.complication_risk[self.risk]
        self.complication = np.random.binomial(1, p, 1)[0]
        if self.complication == 0:
            self.complication_level = 0
        else:
            self.complication_level = np.random.binomial(1, 0.18, 1)[0] + 1

    def choose_refer(self, t_p):

        self.transfer_logic = np.array([[0, 0.31 * t_p, 0.31],
                                        [0, t_p, 1],
                                        [0, 0.41 * t_p, 0.41]])

        if self.facility_delivery.level == 2:
            self.facility_delivery.level = 3
        if self.facility_delivery.idx == 151:
            p = 0
        else:
            p = self.transfer_logic[int(self.facility_delivery.level - 2), self.complication_level]
        self.transfer = np.random.binomial(1, p, 1)[0]

    def choice_referral(self, get_choices, get_probs, coef, facility_objects, l1, l2, df, distance, ref_count, df_births):
        choices, categ = get_choices(self.facility_delivery, l1, l2, df, distance, ref_count, df_births)
        probs = get_probs(choices, categ, self.complication_level, coef)
        options = np.array(choices['Facility'])
        choice = np.random.choice(options, 1, p=probs)[0]
        self.final_deliver = [facility for facility in facility_objects if facility.idx == choice][0]
        self.final_deliver.facility_deliveries += 1
        self.final_deliver.complication_high += (self.complication_level == 2)
        self.final_deliver.complication_low += (self.complication_level == 1)
        self.final_deliver.transferred += (self.transfer == 1)

    def adjust_final(self):
        self.final_deliver = self.facility_delivery

        self.final_deliver.facility_deliveries += 1
        # self.final_deliver.transferred += (self.transfer == 1)
        self.final_deliver.complication_high += (self.complication_level == 2)
        self.final_deliver.complication_low += (self.complication_level == 1)
