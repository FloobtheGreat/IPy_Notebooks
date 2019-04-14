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
u = requests.get(url, headers={'X-DOMO-Developer-Token': '*************************************'})

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

df.to_csv('DOMO_DataFlow_Inputs.csv', index=False)

url = 'https://basspro.domo.com/api/data/v3/datasources/a419bd8d-081c-4d4e-8f1b-2cdf74e6033b/dataversions'

header = { 'content-type':'text/csv', 
          'X-DOMO-Developer-Token': '*************************************'}
p = requests.post(url, 
                 headers = header,
                 data = open(r'DOMO_DataFlow_Inputs.csv', 'rb'))

print(time.time() - t)
