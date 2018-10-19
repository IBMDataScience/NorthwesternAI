import json
import os
from pathlib import Path
from shutil import copyfile
import random
import zipfile

class random_search:


    def __init__(self):

        self.hpo_ranges = {}
        self.hpo_powers = {}
        self.hpo_statics = {}
        self.hyperparameters = []

    def add_hyperparameter(self,name,min,max,step_type,step_value):

        hyperparameter = {
          "name": name
        }

        if isinstance(min, int) and isinstance(max, int):
            value_type = "int_range"
        elif isinstance(min, float) and isinstance(max, float):
            value_type = "double_range"
        else:
            raise Exception("Only int or float values can be provided")

        hyperparameter[value_type] = {
            "min_value": min,
            "max_value": max
        }
        if step_type == "power":
            hyperparameter[value_type][step_type] = step_value
        # else nothing to do

        self.hyperparameters.append(hyperparameter)

    def add_step_range(self,name,min,max,step):

        self.hpo_ranges[name] = [min,max]
        self.add_hyperparameter(name,min,max,"",-1)

    def add_power_range(self,name,min,max,power):
        powers = []
        for index in range(min,max+1):
            powers.append(power**index)
        self.hpo_powers[name] = powers
        self.add_hyperparameter(name,min,max,"power",power)

    def add_static_var(self,name,value):
        # do nothing
        self.hpo_statics[name] = value
        self.add_hyperparameter(name,value,value,"",-1)

    def get_random_hyperparameters(self):

        # add value for ranges
        arguments = {}
        for name in self.hpo_ranges:
            min = self.hpo_ranges[name][0]
            max = self.hpo_ranges[name][1]
            if isinstance(min, int):
                value = random.randint(min,max)
                arguments[name] = value
            else:
                value = random.uniform(min,max)
                arguments[name] = value


        # add value for powers
        for name in self.hpo_powers:
            values = self.hpo_powers[name]
            index = random.randint(0,len(values)-1)
            arguments[name] = values[index]

        # add statics
        for name in self.hpo_statics:
            value = self.hpo_statics[name]
            arguments[name] = value

        return arguments
