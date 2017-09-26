# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:05:06 2017

@author: pairwin
"""

import sys
#sys.path.insert(0, r"C:\users\pairwin\Documents\Github\HelperPI")
sys.path.insert(0, r'/home/pirwin/Git/HelperPI')
import os
import HelperPI
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.externals import joblib
import pyodbc

collist = ['MEDHINC_CY','MEDAGE_CY','CLOSEST_BP','MALES_IN_HOUSHOLD','FEMALES_IN_HOUSHOLD','REWARDS_CUSTOMER',
          'DAYS_AS_CUSTOMER','TOTAL_TRANSACTIONS','REW_TRANSACTIONS','TOTAL_SPEND','DAYS_SINCE_PURCHASE',
          'DAYS_BTW_PURCH']
collist2 = ['MEDAGE_CY','CLOSEST_BP','MALES_IN_HOUSHOLD','FEMALES_IN_HOUSHOLD','REWARDS_CUSTOMER',
          'DAYS_AS_CUSTOMER','TOTAL_SPEND','DAYS_SINCE_PURCHASE']

clf = joblib.load(r'C:\users\pairwin\Documents\GitHub\IPy_Notebooks\Analyses\tabmodelsvc.pkl')

helper = HelperPI.Helper()

path = helper.makeTempDir()
file = 'Scored.csv'
filename = os.path.join(path, file)
helper.deleteTemp(path)

imr = Imputer(missing_values='NaN',strategy='median',axis=0)
stdsc = StandardScaler()

sql = helper.getSQL(r"C:\users\pairwin\Documents\GitHub\IPy_Notebooks\SQL\TAB_SCORING.sql")

#file = helper.readDataToCSV(sql, tmp)

cnxn = pyodbc.connect(r'DRIVER={NetezzaSQL};SERVER=SRVDWHITP01;DATABASE=EDW_SPOKE;UID=pairwin;PWD=pairwin;TIMEOUT=0')

counter = 0
chunksize=100000

for chunk in pd.read_sql(sql, cnxn, chunksize=chunksize):
    counter += chunksize
    print('Working on: ' + str(counter))
    X = chunk[collist2]
    imr = imr.fit(X)
    imputed_data2 = pd.DataFrame(imr.transform(X.values), columns = collist)
    Score_X_std = pd.DataFrame(stdsc.fit_transform(imputed_data2), columns=collist2)
    
    Scored = pd.DataFrame(clf.predict_proba(Score_X_std), columns=['SVM_Score'])
    final = pd.join([chunk, Scored])
    final.to_csv(filename, mode='a')
helper.deleteTemp()