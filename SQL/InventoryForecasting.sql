SELECT DT.FIRST_DAY_OF_WEEK,
	   DT.CALENDAR_QUARTER,
	   CASE WHEN DT.MONTH_NUMBER_IN_CALENDAR_YEAR = 1 THEN 1 ELSE 0 END AS MNTH1,
	   CASE WHEN DT.MONTH_NUMBER_IN_CALENDAR_YEAR = 2 THEN 1 ELSE 0 END AS MNTH2,
	   CASE WHEN DT.MONTH_NUMBER_IN_CALENDAR_YEAR = 3 THEN 1 ELSE 0 END AS MNTH3,
	   CASE WHEN DT.MONTH_NUMBER_IN_CALENDAR_YEAR = 4 THEN 1 ELSE 0 END AS MNTH4,
	   CASE WHEN DT.MONTH_NUMBER_IN_CALENDAR_YEAR = 5 THEN 1 ELSE 0 END AS MNTH5,
	   CASE WHEN DT.MONTH_NUMBER_IN_CALENDAR_YEAR = 6 THEN 1 ELSE 0 END AS MNTH6,
	   CASE WHEN DT.MONTH_NUMBER_IN_CALENDAR_YEAR = 7 THEN 1 ELSE 0 END AS MNTH7,
	   CASE WHEN DT.MONTH_NUMBER_IN_CALENDAR_YEAR = 8 THEN 1 ELSE 0 END AS MNTH8,
	   CASE WHEN DT.MONTH_NUMBER_IN_CALENDAR_YEAR = 10 THEN 1 ELSE 0 END AS MNTH10,
	   CASE WHEN DT.MONTH_NUMBER_IN_CALENDAR_YEAR = 11 THEN 1 ELSE 0 END AS MNTH11,
	   CASE WHEN DT.MONTH_NUMBER_IN_CALENDAR_YEAR = 12 THEN 1 ELSE 0 END AS MNTH12,
	   HOLS.PRIOR_HOLIDAY  AS HOLIDAY_WEEK,
	   CONSENT.UMCSENT,
	   ST.STORE_NUMBER,
	   PD.STYLE_DISPLAY_NUMBER,
	   PD.SKU_DISPLAY_NUMBER,
	   COST.PRICE,
	   --MAX(CASE WHEN PD.SKU_ITEM_LABEL_CODE IN ('B','R') THEN 1 ELSE 0 END) AS Branded,
	   MAX(CURRENT_DATE - PD.SKU_DATE_CREATED) AS SKU_AGE,
	   MAX(CASE WHEN EVENT.PRODUCT_MEMBER_KEY IS NOT NULL THEN 1 ELSE 0 END) AS EVENT_SKU,
	   SUM(SD.INV_QUANTITY) AS TARGET_UNITS
FROM EDW_SPOKE..DIM_DATE DT
	LEFT JOIN EDW_SPOKE..FACT_BPS_SALES_HEADER SH ON DT.DATE_KEY = SH.TRANSACTION_DATE_KEY
	LEFT JOIN EDW_SPOKE..FACT_BPS_SALES_DETAIL SD ON SH.INBOUND_INTERACTION_KEY = SD.SH_INBOUND_INTERACTION_KEY
	LEFT JOIN EDW_SPOKE..DIM_STORE ST ON SD.STORE_MEMBER_KEY = ST.MEMBER_KEY	
	LEFT JOIN EDW_SPOKE..DIM_PRODUCT_BPS PD ON SD.INVENTORY_PRODUCT_MEMBER_KEY = PD.MEMBER_KEY
	LEFT JOIN (
		SELECT PROM.LOCATION_MEMBER_KEY, 
			   PROM.PRODUCT_MEMBER_KEY,
			   PROM.START_DATE,
			   PROM.END_DATE
		FROM EDW_SPOKE..DIM_CAMPAIGN MASTER
			LEFT JOIN EDW_SPOKE..FACT_BPS_PROMOTION PROM ON MASTER.CAMPAIGN_KEY = PROM.CAMPAIGN_KEY	
	) AS EVENT ON ST.MEMBER_KEY = EVENT.LOCATION_MEMBER_KEY
		   AND PD.MEMBER_KEY = EVENT.PRODUCT_MEMBER_KEY
		   AND DT.DATE_VALUE BETWEEN EVENT.START_DATE AND EVENT.END_DATE	
	LEFT JOIN DBM_SANDBOX..CONSUMER_SENTIMENT CONSENT ON DT.MONTH_NUMBER_IN_CALENDAR_YEAR = EXTRACT(MONTH FROM CONSENT.DATE)
														AND DT.CALENDAR_YEAR = EXTRACT(YEAR FROM CONSENT.DATE)
	LEFT JOIN (
		SELECT DT.DATE_VALUE,
			   MAX(CASE WHEN DT.HOLIDAY_INDICATOR = 'Holiday' then 1 else 0 end) OVER(ORDER BY DT.DATE_VALUE DESC ROWS BETWEEN 14 PRECEDING AND 0 FOLLOWING) AS PRIOR_HOLIDAY
		FROM EDW_SPOKE..DIM_DATE DT
		ORDER BY DT.DATE_VALUE
	) HOLS ON DT.DATE_VALUE = HOLS.DATE_VALUE	
	
	LEFT JOIN (
          SELECT 
          P.SKU_DISPLAY_NUMBER,
          P.STYLE_DISPLAY_NUMBER,
          EDW_LANDING..PO_BPS_MRPPCW1P.PRICE AS PRICE,
          EDW_LANDING..PO_BPS_MRPPCW1P.COST AS COST
          FROM
          EDW_LANDING..PO_BPS_MRPPCW1P
          JOIN EDW_SPOKE..DIM_PRODUCT_BPS P
          ON EDW_LANDING..PO_BPS_MRPPCW1P.INUMBR = P.SKU_DISPLAY_NUMBER
        
        
          WHERE PRICECOD = 'USD'
          AND SKU_DISPOSITION_CODE IN ('A','N')
          
    ) COST  ON PD.SKU_DISPLAY_NUMBER = COST.SKU_DISPLAY_NUMBER
	
WHERE ST.CHANNEL_GROUP = 'RETAIL'
	AND DT.DATE_VALUE BETWEEN '2014-01-01' AND '2017-06-01'
	AND SD.RETURN_FLAG <> 'Y'
	AND SD.ACTUAL_SALE_FLAG = 'Y'
    AND PD.SKU_DISPLAY_NUMBER = '214423'
	--AND PD.STYLE_DISPLAY_NUMBER = '52552116'
	AND ST.STORE_NUMBER IN ('001')
GROUP BY DT.FIRST_DAY_OF_WEEK, DT.CALENDAR_QUARTER, DT.MONTH_NUMBER_IN_CALENDAR_YEAR, HOLS.PRIOR_HOLIDAY, 
		 CONSENT.UMCSENT, ST.STORE_NUMBER, PD.STYLE_DISPLAY_NUMBER, PD.SKU_DISPLAY_NUMBER, COST.PRICE
ORDER BY ST.STORE_NUMBER, PD.STYLE_DISPLAY_NUMBER, DT.FIRST_DAY_OF_WEEK
