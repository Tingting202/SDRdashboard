import numpy as np


# FACILIY OBJECTS
class Facility:
    def __init__(self, idx, name, facility_level):
        self.idx = idx
        self.name = name
        self.level = facility_level

        self.facility_deliveries = 0
        self.complication_low = 0
        self.complication_high = 0
        self.transferred = 0

        self.betas_mort = [11.697, -4.889, 132.036, -0.9756, 0.022837, 2.565153]

        self.count = None

    # CALCULATE MORTALITY AT FACILITY LEVEL
    def mortality(self):

        # adjust self transfer
        if (self.complication_low + self.complication_high) > 0:
            self.transferred = self.transferred / (self.complication_low + self.complication_high)
        else:
            self.transferred = 0

        if (self.complication_high == 0) and (self.transferred == 0):
            count = 0
        else:
            value = self.betas_mort[0] + self.betas_mort[1] * self.complication_high + self.betas_mort[
                2] * self.transferred
            pi = 1 / (1 + np.exp(-value))
            lam = np.exp(self.betas_mort[3] + self.betas_mort[4] * self.complication_high + self.betas_mort[
                5] * self.transferred)
            count = (1 - pi) * lam

        self.count = int(count)
