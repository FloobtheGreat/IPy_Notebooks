SELECT
CURRENT_DATE AS DATE_VALUE,
BASE.LAST_NAME||BASE.ADDRESS_LINE_1||BASE.STATE_CODE||BASE.ZIP AS RECORD,
BASE.LAST_NAME,
BASE.ADDRESS_LINE_1,
BASE.ADDRESS_LINE_2,
TRIM(BASE.STATE_CODE) STATE_CODE,
BASE.ZIP,
BASE.MEDHINC_CY,
BASE.MEDAGE_CY,
BASE.CLOSEST_BP AS CLOSEST_BPS,
--BASE.CLOSEST_CAB,
SUM(BASE.MALES_IN_HOUSHOLD) AS MALES_IN_HOUSHOLD,
SUM(BASE.FEMALES_IN_HOUSHOLD) AS FEMALES_IN_HOUSHOLD,
MAX(BASE.REWARDS_CUSTOMER) AS REWARDS_CUSTOMER,
CURRENT_DATE - MIN(SALES.SALE_DATE) AS DAYS_AS_CUSTOMER,
SUM(SALES.TRANSACTIONS) AS TOTAL_TRANSACTIONS,
SUM(CASE WHEN SALES.SALE_DATE < BASE.LAST_TAB_DATE THEN SALES.REW_TRANSACTIONS ELSE 0 END) AS REW_TRANSACTIONS,
SUM(SALES.SALES) AS TOTAL_SPEND,
CURRENT_DATE - MAX(SALES.SALE_DATE) AS DAYS_SINCE_PURCHASE

FROM (

SELECT 

I.LAST_NAME,
AD.ADDRESS_LINE_1,
AD.ADDRESS_LINE_2,
AD.STATE_CODE,
AD.ZIP,
ASD.DISTANCE AS CLOSEST_BP,
ROUND(CZD.CLOSEST_STORE_DIST,2) AS CLOSEST_CAB,
ESR.MEDAGE_CY,
ESR.MEDHINC_CY,
COUNT(DISTINCT CASE WHEN I.GENDER = 'M' THEN I.MEMBER_KEY END) AS MALES_IN_HOUSHOLD,
COUNT(DISTINCT CASE WHEN I.GENDER = 'F' THEN I.MEMBER_KEY END) AS FEMALES_IN_HOUSHOLD,
MAX(CASE WHEN I.REWARDS_ACTIVE_CODE = 'A' THEN 1 ELSE 0 END) AS REWARDS_CUSTOMER,
MAX(M.LAST_TAB_DATE) AS LAST_TAB_DATE,
MIN(I.REWARDS_REGISTRATION_DATE) AS REWARDS_REGISTRATION_DATE

FROM EDW_SPOKE.ETLSVC.DIM_INDIVIDUAL I
JOIN EDW_SPOKE.ADMIN.DIM_ADDRESS AD
ON I.ADDRESS_KEY = AD.MEMBER_KEY
JOIN EDW_SPOKE.ETLSVC.FACT_ADDRESS_STORE_DISTANCE ASD
ON AD.MEMBER_KEY = ASD.ADDRESS_MEMBER_KEY
AND ASD.CLOSEST_STORE_FLAG = 'Y'
LEFT JOIN DBM_SANDBOX.NZ.AAA_CAB_ZIP_DISTANCE CZD
ON AD.ZIP = CZD.ZIP5
LEFT JOIN DBM_SANDBOX.AETURNER.ESRI_ZIP_DATA_1 ESR
ON AD.ZIP = ESR.ID

LEFT JOIN (SELECT BMI.INDIVIDUAL_MEMBER_KEY,
                                                   MAX(DT.DATE_VALUE) AS LAST_TAB_DATE
                                FROM EDW_SPOKE..DIM_DATE DT
                                LEFT JOIN EDW_SPOKE..FACT_MARKETING_INTERACTION FMI ON DT.DATE_KEY = FMI.INTERACTION_DATE_KEY
                                LEFT JOIN EDW_SPOKE..DIM_TREATMENT TR ON FMI.TREATMENT_KEY = TR.TREATMENT_KEY
                                LEFT JOIN EDW_SPOKE..BRIDGE_MARKET_INTER_INDIVIDUAL BMI ON FMI.MARKETING_INTERACTION_KEY = BMI.MARKETING_INTERACTION_KEY 
                                WHERE  TR.INTERACTION_TYPE = 'MAILING'
                                                 AND FMI.HOLDOUT_FLAG = 'N'
                                                AND TR.TYPE_DESCRIPTION IN('RETAIL')
                                                AND TR.CONTENT = 'TAB'
                                                AND BMI.INDIVIDUAL_PRIMARY_TREATMENT = 'Y' 
						AND TR.BOOK_CODE <> 'TH'
						and dt.date_value < CURRENT_DATE
                                GROUP BY BMI.INDIVIDUAL_MEMBER_KEY  
                ) AS M
                ON I.MEMBER_KEY = M.INDIVIDUAL_MEMBER_KEY
                
                GROUP BY 1,2,3,4,5,6,7,8,9

) BASE

LEFT JOIN (SELECT 

SD.SALE_DATE,
I.LAST_NAME,
AD.ADDRESS_LINE_1,
AD.STATE_CODE,
AD.ZIP,
COUNT(DISTINCT SD.SH_INBOUND_INTERACTION_KEY) AS TRANSACTIONS,
COUNT(DISTINCT CASE WHEN REWT.SH_INBOUND_INTERACTION_KEY IS NOT NULL THEN SD.SH_INBOUND_INTERACTION_KEY ELSE NULL END) AS REW_TRANSACTIONS,
SUM(SD.SALES_PRICE) AS SALES

  FROM EDW_SPOKE.ETLSVC.FACT_BPS_SALES_HEADER SH
  JOIN EDW_SPOKE.ETLSVC.FACT_BPS_SALES_DETAIL SD
  ON SH.INBOUND_INTERACTION_KEY = SD.SH_INBOUND_INTERACTION_KEY
  JOIN EDW_SPOKE.ETLSVC.DIM_PERSONA P
  ON SH.CUSTOMER_MEMBER_KEY = P.MEMBER_KEY
  JOIN EDW_SPOKE.ETLSVC.DIM_INDIVIDUAL I
  ON P.INDIVIDUAL_MEMBER_KEY = I.MEMBER_KEY
  JOIN EDW_SPOKE.ADMIN.DIM_ADDRESS AD
  ON I.ADDRESS_KEY = AD.MEMBER_KEY
  JOIN EDW_SPOKE.ETLSVC.DIM_DATE DT
  ON SD.SALE_DATE = DT.DATE_VALUE
  LEFT JOIN (
  	SELECT DISTINCT SH_INBOUND_INTERACTION_KEY
	FROM EDW_SPOKE..FACT_BPS_REWARDS_TRANSACTION
  ) AS REWT ON SH.INBOUND_INTERACTION_KEY = REWT.SH_INBOUND_INTERACTION_KEY
  WHERE SD.ACTUAL_SALE_FLAG = 'Y'
  AND SD.RETURN_FLAG = 'N'
  
  GROUP BY 1,2,3,4,5 ) AS SALES
  
  ON BASE.LAST_NAME = SALES.LAST_NAME
  AND BASE.ADDRESS_LINE_1 = SALES.ADDRESS_LINE_1
  AND BASE.STATE_CODE = SALES.STATE_CODE
  AND BASE.ZIP = SALES.ZIP
  
 
GROUP BY 1,2,3,4,5,6,7,8,9,10

HAVING SUM(SALES.TRANSACTIONS) IS NOT NULL