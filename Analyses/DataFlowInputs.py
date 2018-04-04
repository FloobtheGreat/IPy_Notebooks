# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:24:21 2018

@author: pairwin
"""

import requests
import pandas as pd
import numpy as np
import time

t = time.time()

url = "https://basspro.domo.com/api/dataprocessing/v2/dataflows"
u = requests.get(url, headers={'X-DOMO-Developer-Token': '0af2ea4c1a470d96b02bda43f93d7ca80572d128f1076bf6'})

flows = u.json()

lst = list()

for i in flows['onboardFlows']:
    print(i['name'])
    if 'inputs' in i:
        for j in i['inputs']:
            print('-- ', j['dataSourceName'])
            lst.append([i['name'], j['dataSourceName'], j['dataSourceId']])
    else:
        lst.append([i['name'], 'NULL', 'NULL'])
        
        
df = pd.DataFrame(lst, columns=['DATAFLOW', 'INPUT', 'ID'])



print(time.time() - t)