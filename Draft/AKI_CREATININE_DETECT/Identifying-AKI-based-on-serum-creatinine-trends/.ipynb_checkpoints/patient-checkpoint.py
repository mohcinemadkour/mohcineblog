# -*- coding: utf-8 -*-
"""@author: Mohcine Madkour"""

import eGFR
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Patient(object):
    def __init__(self, mrn, data = {}, age = 0, gender = 0, race = 0):
        self.mrn = mrn
        self.data = DataFrame(Series(data), columns=['value'])
        self.data.index = pd.to_datetime(self.data.index)
        self.crs = self.data.value
        self.age = age
        self.gender = gender
        self.race = race
        if self.crs.size > 0: self.initialize()
        
    def initialize(self):
        self.baseCr = self.__calcBaseCr__()
        self.data['order'] = range(self.crs.size)
        self.data['slope'] = self.__calcSlopes__()
        self.data['peak'] = self.__calcPeaks__()
        self.data['aki'] = self.__calcAKI__()
        self.aki = self.data.value[self.data.aki]
        try: self.egfr = eGFR.ckdepi(self.baseCr, self.age, self.gender, self.race)
        except: self.egfr = np.NaN
        
    def __str__(self): 
        temp = "<MRN {}, Age {}, Gender {}, Race {}, Num CRS {}>"\
            .format(self.mrn, self.age, self.gender, self.race, self.crs.size)
        return temp

    def __repr__(self): return self.__str__()
    
    def __calcBaseCr__(self):
        return np.percentile(self.crs, 25)
    
    def __calcSlopes__(self):
        slopes = []
        for i in range(len(self.crs)-1):
            x1, x2 = self.data.order[i], self.data.order[i+1]
            y1, y2 = self.data.value[i], self.data.value[i+1]
            slopes.append((y2-y1)/(x2-x1))
        # assign slope after last point to the preceding slope
        if len(slopes) == 0: slopes.append(0) 
        else: slopes.append(slopes[len(slopes) - 1])
        return slopes
        
    def __calcPeaks__(self):
        # first point is a peak if following slope is negative
        peaks = [self.data.slope[0] <= 0] 
        for i in range(1, len(self.crs)):
            if self.data.slope[i-1] > 0 and self.data.slope[i] <= 0: 
                peaks.append(True)
            else: peaks.append(False)
        return peaks
    
    def __calcAKI__(self):
        return (self.data.value > self.baseCr * 1.5) & self.data.peak
                               
    def plot(self):
        plt.hlines(self.baseCr, self.data.index.min(), self.data.index.max(), linestyles='dotted')
        plt.plot(self.data.index, self.data.value)        
        plt.plot(self.data.index, self.data.value, 'g.')
        plt.plot(self.data.index[self.data.aki], self.data.value[self.data.aki], 'ro')
        plt.title(str(self))
        plt.xlabel("Date")
        plt.ylabel("Creatinine")