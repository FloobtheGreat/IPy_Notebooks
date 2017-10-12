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
from subprocess import Popen

collist = ['MEDHINC_CY','MEDAGE_CY','CLOSEST_BPS','MALES_IN_HOUSHOLD','FEMALES_IN_HOUSHOLD','REWARDS_CUSTOMER',
          'DAYS_AS_CUSTOMER','TOTAL_TRANSACTIONS','REW_TRANSACTIONS','TOTAL_SPEND','DAYS_SINCE_PURCHASE',
          'DAYS_BTW_PURCH']
collist2 = ['CLOSEST_BPS','MALES_IN_HOUSHOLD','REWARDS_CUSTOMER', 'DAYS_BTW_PURCH',
          'DAYS_AS_CUSTOMER','TOTAL_TRANSACTIONS']

knn = joblib.load(r'C:\users\pairwin\Documents\GitHub\IPy_Notebooks\Analyses\tabmodelknn.pkl')
rf = joblib.load(r'C:\users\pairwin\Documents\GitHub\IPy_Notebooks\Analyses\tabmodelrf.pkl')
et = joblib.load(r'C:\users\pairwin\Documents\GitHub\IPy_Notebooks\Analyses\tabmodelet.pkl')
xgb = joblib.load(r'C:\users\pairwin\Documents\GitHub\IPy_Notebooks\Analyses\tabmodelxgb.pkl')
meta = joblib.load(r'C:\users\pairwin\Documents\GitHub\IPy_Notebooks\Analyses\tabmodelmeta.pkl')
imr = joblib.load(r'C:\users\pairwin\Documents\GitHub\IPy_Notebooks\Analyses\tabmodel_impute.pkl')
log = joblib.load(r'C:\users\pairwin\Documents\GitHub\IPy_Notebooks\Analyses\tabmodellog.pkl')

helper = HelperPI.Helper()

path = helper.makeTempDir()
file = 'Scored.csv'
filename = os.path.join(path, file)
#helper.deleteTemp(path)


stdsc = StandardScaler()

sql = helper.getSQL(r"C:\users\pairwin\Documents\GitHub\IPy_Notebooks\SQL\TAB_SCORING.sql")

#file = helper.readDataToCSV(sql, tmp)

cnxn = pyodbc.connect(r'DRIVER={NetezzaSQL};SERVER=SRVDWHITP01;DATABASE=DBM_SANDBOX;UID=pairwin;PWD=pairwin;TIMEOUT=0')

counter = 0
chunksize=100000




for chunk in pd.read_sql(sql, cnxn, chunksize=chunksize):
    chunk['DAYS_BTW_PURCH'] = chunk['DAYS_AS_CUSTOMER']/chunk['TOTAL_TRANSACTIONS']
    counter += chunksize
    print('Working on: ' + str(counter))
    X = chunk[collist2]
    
    base = chunk[['LAST_NAME','ADDRESS_LINE_1','ADDRESS_LINE_2','STATE_CODE','ZIP']]
    imputed_data2 = pd.DataFrame(imr.transform(X.values), columns = collist2)

    Score_X_std = stdsc.fit_transform(imputed_data2.values)
    
    #knnScored = knn.predict_proba(Score_X_std)
    #knnScored_pos = pd.DataFrame(knnScored[:,1], columns = ['knn_Score'])
    
    #rfScored = rf.predict_proba(Score_X_std)
    #rfScored_pos = pd.DataFrame(rfScored[:,1], columns = ['rf_Score'])
    
    #etScored = et.predict_proba(Score_X_std)
    #etScored_pos = pd.DataFrame(etScored[:,1], columns = ['et_Score'])
    
#    xgbScored = xgb.predict_proba(Score_X_std)
#    xgbScored_pos = pd.DataFrame(xgbScored[:,1], columns = ['xgb_Score'])
    
    #objs = [knnScored_pos, rfScored_pos, etScored_pos]
    
    #toScore = pd.concat(objs=objs, axis=1, join_axes=[X.index])
    
    metaScored = meta.predict_proba(Score_X_std)
    metaScored_pos = pd.DataFrame(metaScored[:,1], columns = ['ENS_Score'])
    #logScored = log.predict_proba(Score_X_std)
    #logScored_pos = pd.DataFrame(logScored[:,1], columns = ['log_Score'])
    #Scores = toScore.join(metaScored_pos)
    final = base.join(metaScored_pos)
    final.to_csv(filename, mode='a', index=False)
    
    
    
new_sql = r"""DROP TABLE DBM_SANDBOX..PI_TABMODEL_BOOSTED IF EXISTS;

CREATE TABLE DBM_SANDBOX..PI_TABMODEL_BOOSTED( 
     LAST_NAME CHARACTER VARYING(30),
     ADDRESS_LINE_1 CHARACTER VARYING(30),
     ADDRESS_LINE_2 CHARACTER VARYING(30),
     STATE_CODE CHARACTER VARYING(19),
     ZIP CHARACTER VARYING(5),
     ENS_SCORE DOUBLE)
     DISTRIBUTE ON (LAST_NAME, ADDRESS_LINE_1, STATE_CODE, ZIP);


INSERT INTO DBM_SANDBOX..PI_TABMODEL_BOOSTED
     SELECT * FROM 
     EXTERNAL '""" + filename + """' 
     USING (
      DELIMITER ','
      Y2BASE 2000
      ENCODING 'internal'
      REMOTESOURCE 'ODBC'
      ESCAPECHAR '\'
      MAXERRORS 0
); """
     
with open(r'C:\Users\pairwin\Documents\GitHub\IPy_Notebooks\Analyses\newsql.sql', 'w') as tfile:
    print(new_sql, file=tfile)          

p = Popen(r"C:\Users\pairwin\Documents\GitHub\IPy_Notebooks\Analyses\ScoredSql.bat")
stdout, stderr = p.communicate()

cnxn.close()
    
helper.deleteTemp(path)