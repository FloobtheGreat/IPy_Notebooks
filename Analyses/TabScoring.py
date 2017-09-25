# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:05:06 2017

@author: pairwin
"""

import sys
sys.path.insert(0, r"C:\users\pairwin\Documents\Github\HelperPI")

import HelperPI
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.externals import joblib




helper = HelperPI.Helper()



tmp = helper.makeTempDir()
#helper.deleteTemp(tmp)

sql = helper.getSQL(r"C:\users\pairwin\Documents\GitHub\IPy_Notebooks\SQL\TAB_SCORING.sql")

file = helper.readDataToCSV(sql, tmp)



#base, dtypesdf = helper.readData(sql)
#base = pd.read_csv(r'C:\users\pairwin\Documents\GitHub\IPy_Notebooks\SQL\tab_model.csv', parse_dates=['DATE_VALUE'])
#base = pd.read_csv(r'/home/pirwin/Git/IPy_Notebooks/SQL/tab_model.csv',parse_dates=['DATE_VALUE'])
#dtypes = helper.getDtypes(base)

collist = ['MEDHINC_CY','MEDAGE_CY','CLOSEST_BP','MALES_IN_HOUSHOLD','FEMALES_IN_HOUSHOLD','REWARDS_CUSTOMER',
          'DAYS_AS_CUSTOMER','TOTAL_TRANSACTIONS','REW_TRANSACTIONS','TOTAL_SPEND','DAYS_SINCE_PURCHASE',
          'DAYS_BTW_PURCH']
X = base[collist]


missing_df = X.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['missing_ratio'] = missing_df['missing_count'] / X.shape[0]
missing_df.loc[missing_df['missing_ratio']>0.01]

imr = Imputer(missing_values='NaN',strategy='median',axis=0)
imr = imr.fit(X)
imputed_data = pd.DataFrame(imr.transform(X.values), columns = collist)


collist2 = ['MEDAGE_CY','CLOSEST_BP','MALES_IN_HOUSHOLD','FEMALES_IN_HOUSHOLD','REWARDS_CUSTOMER',
          'DAYS_AS_CUSTOMER','TOTAL_SPEND','DAYS_SINCE_PURCHASE']

imputed_data2 = imputed_data[collist2]


stdsc = StandardScaler()
Score_X_std = pd.DataFrame(stdsc.fit_transform(imputed_data2), columns=collist2)
clf = joblib.load(r'C:\users\pairwin\Documents\GitHub\IPy_Notebooks\Analyses\tabmodelsvc.pkl')
Scored = clf.predict_proba(Score_X_std)