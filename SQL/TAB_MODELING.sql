SELECT *
FROM (
SELECT CURRENT_DATE AS DATE_VALUE,
	   BASE.LAST_NAME||BASE.ADDRESS_LINE_1||BASE.STATE_CODE||BASE.ZIP AS RECORD,
	   BASE.MEDHINC_CY,
	   BASE.MEDAGE_CY,
	   BASE.CLOSEST_BP,
	   BASE.CLOSEST_CAB,
	   BASE.MALES_IN_HOUSHOLD,
	   BASE.FEMALES_IN_HOUSHOLD,
	   MAX(CASE WHEN BASE.REWARDS_REGISTRATION_DATE < BASE.LAST_TAB_DATE THEN 1 ELSE 0 END) AS REWARDS_CUSTOMER,
	   (CURRENT_DATE - 365) - MIN(CASE WHEN SALES.SALE_DATE < BASE.LAST_TAB_DATE THEN SALES.SALE_DATE END) AS DAYS_AS_CUSTOMER,
	   CAST(SUM(CASE WHEN SALES.SALE_DATE < BASE.LAST_TAB_DATE THEN SALES.TRANSACTIONS END) AS FLOAT) AS TOTAL_TRANSACTIONS,
	   CAST(SUM(CASE WHEN SALES.SALE_DATE < BASE.LAST_TAB_DATE THEN SALES.REW_TRANSACTIONS END) AS FLOAT) AS REW_TRANSACTIONS,
	   SUM(CASE WHEN SALES.SALE_DATE < BASE.LAST_TAB_DATE THEN SALES.SALES END) AS TOTAL_SPEND,
	   (CURRENT_DATE - 365) - MAX(CASE WHEN SALES.SALE_DATE < BASE.LAST_TAB_DATE THEN SALES.SALE_DATE END) AS DAYS_SINCE_PURCHASE,
	   SUM(CASE WHEN SALES.SALE_DATE > BASE.LAST_TAB_DATE AND SALES.SALE_DATE <= BASE.LAST_TAB_DATE + 15 THEN SALES.SALES ELSE 0 END) AS TARGET_VALUE15,
	   MAX(CASE WHEN SALES.SALE_DATE > BASE.LAST_TAB_DATE AND SALES.SALE_DATE <= BASE.LAST_TAB_DATE + 15 THEN 1 ELSE 0 END) AS TARGET_PURCH_NEXT15
FROM (
	SELECT I.LAST_NAME,
		   AD.ADDRESS_LINE_1,
		   GEO.STATE_CODE,
		   GEO.ZIP,
		   ASD.DISTANCE AS CLOSEST_BP,
		   ROUND(CZD.CLOSEST_STORE_DIST,2) AS CLOSEST_CAB,
		   ESR.MEDAGE_CY,
		   ESR.MEDHINC_CY,
		   COUNT(DISTINCT CASE WHEN I.GENDER = 'M' THEN I.MEMBER_KEY END) AS MALES_IN_HOUSHOLD,
		   COUNT(DISTINCT CASE WHEN I.GENDER = 'F' THEN I.MEMBER_KEY END) AS FEMALES_IN_HOUSHOLD,
		   MAX(CASE WHEN I.REWARDS_ACTIVE_CODE = 'A' THEN 1 ELSE 0 END) AS REWARDS_CUSTOMER,
		   MAX(M.LAST_TAB_DATE) AS LAST_TAB_DATE,
		   MIN(I.REWARDS_REGISTRATION_DATE) AS REWARDS_REGISTRATION_DATE
	FROM EDW_SPOKE..DIM_INDIVIDUAL I
	 JOIN EDW_SPOKE..DIM_ADDRESS AD ON I.ADDRESS_KEY = AD.MEMBER_KEY
	 JOIN EDW_SPOKE..DIM_GEOGRAPHY GEO ON AD.GEOGRAPHY_MEMBER_KEY = GEO.MEMBER_KEY
	 JOIN EDW_SPOKE..FACT_ADDRESS_STORE_DISTANCE ASD ON AD.MEMBER_KEY = ASD.ADDRESS_MEMBER_KEY
	 												   AND ASD.CLOSEST_STORE_FLAG = 'Y'
	 LEFT JOIN DBM_SANDBOX..AAA_CAB_ZIP_DISTANCE CZD ON AD.ZIP = CZD.ZIP5
	 LEFT JOIN DBM_SANDBOX..ESRI_ZIP_DATA_1 ESR ON AD.ZIP = ESR.ID
	 LEFT JOIN (
	 	SELECT BMI.INDIVIDUAL_MEMBER_KEY,
			   MAX(DT.DATE_VALUE) AS LAST_TAB_DATE
		FROM EDW_SPOKE..DIM_DATE DT
		LEFT JOIN EDW_SPOKE..FACT_MARKETING_INTERACTION FMI ON DT.DATE_KEY = FMI.INTERACTION_DATE_KEY
		LEFT JOIN EDW_SPOKE..DIM_TREATMENT TR ON FMI.TREATMENT_KEY = TR.TREATMENT_KEY
		LEFT JOIN EDW_SPOKE..BRIDGE_MARKET_INTER_INDIVIDUAL BMI ON FMI.MARKETING_INTERACTION_KEY = BMI.MARKETING_INTERACTION_KEY 
		WHERE  TR.INTERACTION_TYPE = 'MAILING'
		   	 AND FMI.HOLDOUT_FLAG = 'N'
		 	 AND TR.TYPE_DESCRIPTION IN('RETAIL')
		 	 AND TR.CONTENT = 'TAB'
			 AND DT.DATE_VALUE < CURRENT_DATE - 365
		 	 AND BMI.INDIVIDUAL_PRIMARY_TREATMENT = 'Y' 
			 AND TR.BOOK_CODE <> 'TH'
		GROUP BY BMI.INDIVIDUAL_MEMBER_KEY  
	 ) AS M ON I.MEMBER_KEY = M.INDIVIDUAL_MEMBER_KEY
	GROUP BY 1,2,3,4,5,6,7,8
) BASE
LEFT JOIN (
	SELECT SD.SALE_DATE,
		   I.LAST_NAME,
		   AD.ADDRESS_LINE_1,
		   AD.STATE_CODE,
		   AD.ZIP,
		   COUNT(DISTINCT SD.SH_INBOUND_INTERACTION_KEY) AS TRANSACTIONS,
		   COUNT(DISTINCT CASE WHEN REWT.SH_INBOUND_INTERACTION_KEY IS NOT NULL THEN SD.SH_INBOUND_INTERACTION_KEY ELSE NULL END) AS REW_TRANSACTIONS,
		   SUM(SD.SALES_PRICE) AS SALES
	FROM EDW_SPOKE..FACT_BPS_SALES_HEADER SH
	 JOIN EDW_SPOKE..FACT_BPS_SALES_DETAIL SD ON SH.INBOUND_INTERACTION_KEY = SD.SH_INBOUND_INTERACTION_KEY
	 JOIN EDW_SPOKE..DIM_PERSONA P ON SH.CUSTOMER_MEMBER_KEY = P.MEMBER_KEY
	 JOIN EDW_SPOKE..DIM_INDIVIDUAL I ON P.INDIVIDUAL_MEMBER_KEY = I.MEMBER_KEY
	 JOIN EDW_SPOKE..DIM_ADDRESS AD ON I.ADDRESS_KEY = AD.MEMBER_KEY
	 JOIN EDW_SPOKE..DIM_DATE DT ON SD.SALE_DATE = DT.DATE_VALUE
	 LEFT JOIN (
	 	SELECT DISTINCT SH_INBOUND_INTERACTION_KEY
		FROM EDW_SPOKE..FACT_BPS_REWARDS_TRANSACTION
	 ) AS REWT ON SH.INBOUND_INTERACTION_KEY = REWT.SH_INBOUND_INTERACTION_KEY
	 WHERE SD.ACTUAL_SALE_FLAG = 'Y'
		AND SD.RETURN_FLAG = 'N'
	 GROUP BY 1,2,3,4,5 
) AS SALES ON BASE.LAST_NAME = SALES.LAST_NAME
	        AND BASE.ADDRESS_LINE_1 = SALES.ADDRESS_LINE_1
			AND BASE.STATE_CODE = SALES.STATE_CODE
			AND BASE.ZIP = SALES.ZIP
WHERE BASE.LAST_TAB_DATE IS NOT NULL
GROUP BY BASE.LAST_NAME, BASE.ADDRESS_LINE_1, BASE.STATE_CODE, BASE.ZIP, BASE.MEDHINC_CY, BASE.MEDAGE_CY, 
	     BASE.CLOSEST_BP, BASE.CLOSEST_CAB, BASE.MALES_IN_HOUSHOLD, BASE.FEMALES_IN_HOUSHOLD

HAVING SUM(SALES.TRANSACTIONS) IS NOT NULL
	AND MIN(SALES.SALE_DATE) < MAX(BASE.LAST_TAB_DATE)
) FIN
ORDER BY RANDOM()
LIMIT 100000