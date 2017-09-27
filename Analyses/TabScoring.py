# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:05:06 2017

@author: pairwin
"""

import sys
sys.path.insert(0, r"C:\users\pairwin\Documents\Github\HelperPI")
#sys.path.insert(0, r'/home/pirwin/Git/HelperPI')
import os
import HelperPI
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import pyodbc

collist = ['MEDHINC_CY','MEDAGE_CY','CLOSEST_BPS','MALES_IN_HOUSHOLD','FEMALES_IN_HOUSHOLD','REWARDS_CUSTOMER',
          'DAYS_AS_CUSTOMER','TOTAL_TRANSACTIONS','REW_TRANSACTIONS','TOTAL_SPEND','DAYS_SINCE_PURCHASE',
          'DAYS_BTW_PURCH']
collist2 = ['MEDAGE_CY','CLOSEST_BPS','MALES_IN_HOUSHOLD','FEMALES_IN_HOUSHOLD','REWARDS_CUSTOMER',
          'DAYS_AS_CUSTOMER','TOTAL_SPEND','DAYS_SINCE_PURCHASE']

clf = joblib.load(r'C:\users\pairwin\Documents\GitHub\IPy_Notebooks\Analyses\tabmodelsvc.pkl')
imr = joblib.load(r'C:\users\pairwin\Documents\GitHub\IPy_Notebooks\Analyses\tabmodel_impute.pkl')

helper = HelperPI.Helper()

path = helper.makeTempDir()
file = 'Scored.csv'
filename = os.path.join(path, file)
#helper.deleteTemp(path)


stdsc = StandardScaler()

sql = helper.getSQL(r"C:\users\pairwin\Documents\GitHub\IPy_Notebooks\SQL\TAB_SCORING.sql")

#file = helper.readDataToCSV(sql, tmp)

cnxn = pyodbc.connect(r'DRIVER={NetezzaSQL};SERVER=SRVDWHITP01;DATABASE=EDW_SPOKE;UID=pairwin;PWD=pairwin;TIMEOUT=0')

counter = 0
chunksize=100000



for chunk in pd.read_sql(sql, cnxn, chunksize=chunksize):
    chunk['DAYS_BTW_PURCH'] = chunk['DAYS_AS_CUSTOMER']/chunk['TOTAL_TRANSACTIONS']
    counter += chunksize
    print('Working on: ' + str(counter))
    X = chunk[collist]
    base = chunk[['LAST_NAME','ADDRESS_LINE_1','ADDRESS_LINE_2','STATE_CODE','ZIP']]
    imputed_data2 = pd.DataFrame(imr.transform(X.values), columns = collist)
    
    imputed_data2 = imputed_data2[collist2]
    Score_X_std = stdsc.fit_transform(imputed_data2.values)
    
    Scored = clf.predict_proba(Score_X_std)
    Scored_pos = pd.DataFrame(Scored[:,1], columns = ['SVM_Score'])
    final = base.join([Scored_pos])
    final.to_csv(filename, mode='a')
    
    
helper.deleteTemp(path)