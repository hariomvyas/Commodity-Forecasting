# Commodity-Forecasting

To forecast the future price of commodities on the US market using technical analysis indicators, a machine learning model was developed and put into use in this article. For this, data on the commodities Crude Oil, Silver, Platinum, Palladium, Rhodium, and Oil were obtained from several sources that provide real-time data. These data were then cleaned, pre-processed, and divided. Machine learning is employed to do the data perdition and forecast the price of gold for the future from the pre-processed data. However, based on the outcomes, it can be said that the constructed model achieved a good prediction performance for all the examined commodities.

## About Data

Data for this study is collected from November 18th 2011 to January 1st 2019 from various sources. The data has 1718 rows in total and 80 columns in total. Data for attributes, such as Oil Price, Standard and Poor’s (S&P) 500 index, Dow Jones Index US Bond rates (10 years), Euro USD exchange rates, prices of precious metals Silver and Platinum and other metals such as Palladium and Rhodium, prices of US Dollar Index, Eldorado Gold Corporation and Gold Miners ETF were gathered.

**Attributes:**

**Features:**


* Gold ETF :- Date, Open, High, Low, Close and Volume.
* S&P 500 Index :- 'SP_open', 'SP_high', 'SP_low', 'SP_close', 'SP_Ajclose', 'SP_volume'
* Dow Jones Index :- 'DJ_open','DJ_high', 'DJ_low', 'DJ_close', 'DJ_Ajclose', 'DJ_volume'
* Eldorado Gold Corporation (EGO) :- 'EG_open', 'EG_high', 'EG_low', 'EG_close', 'EG_Ajclose', 'EG_volume'
* EURO - USD Exchange Rate :- 'EU_Price','EU_open', 'EU_high', 'EU_low', 'EU_Trend'
* Brent Crude Oil Futures :- 'OF_Price', 'OF_Open', 'OF_High', 'OF_Low', 'OF_Volume', 'OF_Trend'
* Crude Oil WTI USD :- 'OS_Price', 'OS_Open', 'OS_High', 'OS_Low', 'OS_Trend'
* Silver Futures :- 'SF_Price', 'SF_Open', 'SF_High', 'SF_Low', 'SF_Volume', 'SF_Trend'
* US Bond Rate (10 years) :- 'USB_Price', 'USB_Open', 'USB_High','USB_Low', 'USB_Trend'
* Platinum Price :- 'PLT_Price', 'PLT_Open', 'PLT_High', 'PLT_Low','PLT_Trend'
* Palladium Price :- 'PLD_Price', 'PLD_Open', 'PLD_High', 'PLD_Low','PLD_Trend'
* Rhodium Prices :- 'RHO_PRICE'
* US Dollar Index : 'USDI_Price', 'USDI_Open', 'USDI_High','USDI_Low', 'USDI_Volume', 'USDI_Trend'
* Gold Miners ETF :- 'GDX_Open', 'GDX_High', 'GDX_Low', 'GDX_Close', 'GDX_Adj Close', 'GDX_Volume'
* Oil ETF USO :- 'USO_Open','USO_High', 'USO_Low', 'USO_Close', 'USO_Adj Close', 'USO_Volume'

**Target Variable**

* Gold ETF :- Adjusted Close

## Methodology

![image](https://user-images.githubusercontent.com/49945980/205475595-b6ec8f1f-19a1-4b01-82ec-35b6fae41efe.png)

## Results

![image](https://user-images.githubusercontent.com/49945980/205475634-fe8ed0ba-d9c3-4233-9a31-da9a27295620.png)

![image](https://user-images.githubusercontent.com/49945980/205475642-86d1a91f-9110-4068-a76d-a3a79a9aceea.png)


# Team

*   Hariom Vyas
*   Krutal Patel
*   Viral Jani
*   Durga Siva Sai Verma Rudraraju
