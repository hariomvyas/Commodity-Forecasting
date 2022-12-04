import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

header = st.container()
dataset = st.container()
train_test_split = st.container()
features = st.container()
model_training = st.container()
with header:
    st.title('Commodity Forecasting using Machine Learning')
    st.markdown("In the past, gold had been used as currency in several countries, including the USA. To ensure the repayment of foreign loans and to limit inflation, which has the effect of revealing a country's financial strength, precious metals like gold are now kept by the central banks of all nations.")
    st.markdown("Investors may choose the best time to purchase (or sell) gold by predicting the increase and decrease in the daily gold rates.")
    st.markdown("In this project, we would use the most complete collection of characteristics to estimate gold rates. We would also use a variety of machine learning methods and compare the outcomes. We also pinpoint the characteristics that have a significant impact on gold rates.")
with dataset:
    st.header('Historical Gold Price dataset')
    st.markdown("Data for this study is collected from November 18th 2011 to January 1st 2019 from various sources. The data has 1718 rows in total and 80 columns in total. Data for attributes, such as Oil Price, Standard and Poor’s (S&P) 500 index, Dow Jones Index US Bond rates (10 years), Euro USD exchange rates, prices of precious metals Silver and Platinum and other metals such as Palladium and Rhodium, prices of US Dollar Index, Eldorado Gold Corporation and Gold Miners ETF were gathered.")
    st.subheader("Attributes")
    st.markdown("**Features:**")
    st.markdown("* Gold ETF :- Date, Open, High, Low, Close and Volume.")
    st.markdown("* S&P 500 Index :- 'SP_open', 'SP_high', 'SP_low', 'SP_close', 'SP_Ajclose', 'SP_volume'")
    st.markdown("* Dow Jones Index :- 'DJ_open','DJ_high', 'DJ_low', 'DJ_close', 'DJ_Ajclose', 'DJ_volume'")
    st.markdown("* Eldorado Gold Corporation (EGO) :- 'EG_open', 'EG_high', 'EG_low', 'EG_close', 'EG_Ajclose', 'EG_volume'")
    st.markdown("* EURO - USD Exchange Rate :- 'EU_Price','EU_open', 'EU_high', 'EU_low', 'EU_Trend'")
    st.markdown("* Brent Crude Oil Futures :- 'OF_Price', 'OF_Open', 'OF_High', 'OF_Low', 'OF_Volume', 'OF_Trend'")
    st.markdown("* Crude Oil WTI USD :- 'OS_Price', 'OS_Open', 'OS_High', 'OS_Low', 'OS_Trend'")
    st.markdown("* Silver Futures :- 'SF_Price', 'SF_Open', 'SF_High', 'SF_Low', 'SF_Volume', 'SF_Trend'")
    st.markdown("* US Bond Rate (10 years) :- 'USB_Price', 'USB_Open', 'USB_High','USB_Low', 'USB_Trend'")
    st.markdown("* Platinum Price :- 'PLT_Price', 'PLT_Open', 'PLT_High', 'PLT_Low','PLT_Trend'")
    st.markdown("* Palladium Price :- 'PLD_Price', 'PLD_Open', 'PLD_High', 'PLD_Low','PLD_Trend'")
    st.markdown("* Rhodium Prices :- 'RHO_PRICE'")
    st.markdown("* US Dollar Index : 'USDI_Price', 'USDI_Open', 'USDI_High','USDI_Low', 'USDI_Volume', 'USDI_Trend'")
    st.markdown("* Gold Miners ETF :- 'GDX_Open', 'GDX_High', 'GDX_Low', 'GDX_Close', 'GDX_Adj Close', 'GDX_Volume'")
    st.markdown("* Oil ETF USO :- 'USO_Open','USO_High', 'USO_Low', 'USO_Close', 'USO_Adj Close', 'USO_Volume'")

    st.markdown("**Target Variable:**")
    st.markdown("* Gold ETF :- Adjusted Close")

    DATA_URL = ("https://raw.githubusercontent.com/hariomvyas/Commodity-Forecasting/main/Data/FINAL_USO.csv")

    @st.cache(allow_output_mutation = True)
    def load_data():
        data = pd.read_csv(DATA_URL,na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=True)
        return data
    
    st.markdown("**Data Preview**")
    data = load_data()
    st.write(data.head())
    column1 = st.selectbox("which column 1 do you want to choose? ", options = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SP_open',
       'SP_high', 'SP_low', 'SP_close', 'SP_Ajclose', 'SP_volume', 'DJ_open',
       'DJ_high', 'DJ_low', 'DJ_close', 'DJ_Ajclose', 'DJ_volume', 'EG_open',
       'EG_high', 'EG_low', 'EG_close', 'EG_Ajclose', 'EG_volume', 'EU_Price',
       'EU_open', 'EU_high', 'EU_low', 'EU_Trend', 'OF_Price', 'OF_Open',
       'OF_High', 'OF_Low', 'OF_Volume', 'OF_Trend', 'OS_Price', 'OS_Open',
       'OS_High', 'OS_Low', 'OS_Trend', 'SF_Price', 'SF_Open', 'SF_High',
       'SF_Low', 'SF_Volume', 'SF_Trend', 'USB_Price', 'USB_Open', 'USB_High',
       'USB_Low', 'USB_Trend', 'PLT_Price', 'PLT_Open', 'PLT_High', 'PLT_Low',
       'PLT_Trend', 'PLD_Price', 'PLD_Open', 'PLD_High', 'PLD_Low',
       'PLD_Trend', 'RHO_PRICE', 'USDI_Price', 'USDI_Open', 'USDI_High',
       'USDI_Low', 'USDI_Volume', 'USDI_Trend', 'GDX_Open', 'GDX_High',
       'GDX_Low', 'GDX_Close', 'GDX_Adj Close', 'GDX_Volume', 'USO_Open',
       'USO_High', 'USO_Low', 'USO_Close', 'USO_Adj Close', 'USO_Volume'], index = 0)
    
    data2  = data[column1]

    df_p = pd.DataFrame({column1:data2})
    fig = px.line(df_p)
    st.write(fig)

    
with train_test_split:  
    def compute_daily_returns(df):
        daily_return = (df / df.shift(1)) - 1
        daily_return[0] = 0
        return daily_return
    feature_columns = ['Open', 'High', 'Low', 'Volume','SP_open','SP_high','SP_low','SP_Ajclose','SP_volume','DJ_open','DJ_high', 'DJ_low',  'DJ_Ajclose', 'DJ_volume', 'EG_open','EG_high', 'EG_low',  
                   'EG_Ajclose', 'EG_volume', 'EU_Price','EU_open', 'EU_high', 'EU_low', 'EU_Trend', 'OF_Price','OF_Open','OF_High', 'OF_Low', 'OF_Volume', 'OF_Trend', 'OS_Price', 'OS_Open','OS_High', 'OS_Low', 'OS_Trend', 'SF_Price', 'SF_Open', 'SF_High',
                   'SF_Low', 'SF_Volume', 'SF_Trend', 'USB_Price', 'USB_Open', 'USB_High','USB_Low', 'USB_Trend', 'PLT_Price', 'PLT_Open', 'PLT_High', 'PLT_Low',
                    'PLT_Trend', 'PLD_Price', 'PLD_Open', 'PLD_High', 'PLD_Low','PLD_Trend', 'RHO_PRICE', 'USDI_Price', 'USDI_Open', 'USDI_High',
                     'USDI_Low', 'USDI_Volume', 'USDI_Trend','GDX_Open', 'GDX_High',
       'GDX_Low', 'GDX_Close', 'GDX_Adj Close', 'GDX_Volume', 'USO_Open',
       'USO_High', 'USO_Low', 'USO_Close', 'USO_Adj Close', 'USO_Volume','SMA', 'Upper_band', 'Lower_band', 'DIF', 'MACD','RSI','STDEV','Open_Close', 'High_Low']
    GLD_adj_close = data['Adj Close']
    SPY_adj_close = data['SP_Ajclose']
    DJ_adj_close  = data['DJ_Ajclose']
    EG_adj_close =  data['EG_Ajclose']
    USO_Adj_close = data['USO_Adj Close']
    GDX_Adj_close = data['GDX_Adj Close']
    EU_price      = data['EU_Price']
    OF_price      = data['OF_Price']
    OS_price      = data['OS_Price']
    SF_price      = data['SF_Price']
    USB_price      = data['USB_Price']
    PLT_price      = data['PLT_Price']
    PLD_price      = data['PLD_Price']
    rho_price      = data['RHO_PRICE']
    usdi_price      = data['USDI_Price']



    GLD_daily_return = compute_daily_returns(GLD_adj_close)
    SPY_daily_return = compute_daily_returns(SPY_adj_close)
    DJ_adj_return    = compute_daily_returns(DJ_adj_close)
    EG_adj_return     = compute_daily_returns(EG_adj_close)
    USO_Adj_return    = compute_daily_returns(USO_Adj_close)
    GDX_Adj_return   =compute_daily_returns(GDX_Adj_close)
    EU_return        = compute_daily_returns(EU_price)
    OF_price         =compute_daily_returns(OF_price)
    OS_price         =compute_daily_returns(OS_price)
    SF_price         =compute_daily_returns(SF_price)
    USB_price         =compute_daily_returns(USB_price)
    PLT_price         =compute_daily_returns(PLT_price)
    PLD_price         =compute_daily_returns(PLD_price)
    rho_price         =compute_daily_returns(rho_price)
    USDI_price         =compute_daily_returns(usdi_price)

    
    def calculate_MACD(df, nslow=26, nfast=12):
        emaslow = df.ewm(span=nslow, min_periods=nslow, adjust=True, ignore_na=False).mean()
        emafast = df.ewm(span=nfast, min_periods=nfast, adjust=True, ignore_na=False).mean()
        dif = emafast - emaslow
        MACD = dif.ewm(span=9, min_periods=9, adjust=True, ignore_na=False).mean()
        return dif, MACD

    def calculate_RSI(df, periods=14):
        # wilder's RSI
        delta = df.diff()
        up, down = delta.copy(), delta.copy()

        up[up < 0] = 0
        down[down > 0] = 0

        rUp = up.ewm(com=periods,adjust=False).mean()
        rDown = down.ewm(com=periods, adjust=False).mean().abs()

        rsi = 100 - 100 / (1 + rUp / rDown)
        return rsi

    def calculate_SMA(df, peroids=15):
        SMA = df.rolling(window=peroids, min_periods=peroids, center=False).mean()
        return SMA

    def calculate_BB(df, peroids=15):
        STD = df.rolling(window=peroids,min_periods=peroids, center=False).std()
        SMA = calculate_SMA(df)
        upper_band = SMA + (2 * STD)
        lower_band = SMA - (2 * STD)
        return upper_band, lower_band

    def calculate_stdev(df,periods=5):
        STDEV = df.rolling(periods).std()
        return STDEV
        
    SMA_GLD = calculate_SMA(GLD_adj_close)
    # Calculate Bollinger Bands for GLD
    upper_band, lower_band = calculate_BB(GLD_adj_close)
    # Calculate MACD for GLD
    DIF, MACD = calculate_MACD(GLD_adj_close)
    # Calculate RSI for GLD
    RSI = calculate_RSI(GLD_adj_close)
    # Calculating Standard deviation for GLD
    STDEV= calculate_stdev(GLD_adj_close)
    Open_Close=data.Open - data.Close
    High_Low=data.High-data.Low
    
    
    test = data
    test['SMA'] = SMA_GLD
    test['Upper_band'] = upper_band
    test['Lower_band'] = lower_band
    test['DIF'] = DIF
    test['MACD'] = MACD
    test['RSI'] = RSI
    test['STDEV'] = STDEV
    test['Open_Close']=Open_Close
    test['High_Low']=High_Low


    # Dropping first 33 records from the data as it has null values because of introduction of technical indicators
    test = test[33:]

    # Target column
    target_adj_close = pd.DataFrame(test['Adj Close'])


   
    scaler = MinMaxScaler()
    feature_minmax_transform_data = scaler.fit_transform(test[feature_columns])
    feature_minmax_transform = pd.DataFrame(columns=feature_columns, data=feature_minmax_transform_data, index=test.index)
    feature_minmax_transform.head()
    

    # Shift target array because we want to predict the n + 1 day value


    target_adj_close = target_adj_close.shift(-1)
    validation_y = target_adj_close[-90:-1]
    target_adj_close = target_adj_close[:-90]

    # Taking last 90 rows of data to be validation set
    validation_X = feature_minmax_transform[-90:-1]
    feature_minmax_transform = feature_minmax_transform[:-90]
   

   
    ts_split= TimeSeriesSplit(n_splits=10)
    for train_index, test_index in ts_split.split(feature_minmax_transform):
        X_train, X_test = feature_minmax_transform[:len(train_index)], feature_minmax_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = target_adj_close[:len(train_index)].values.ravel(), target_adj_close[len(train_index): (len(train_index)+len(test_index))].values.ravel()
    def validate_result(model, model_name):
        fig, ax = plt.subplots()
        predicted = model.predict(validation_X)
        RSME_score = np.sqrt(mean_squared_error(validation_y, predicted))
        
        
        R2_score = r2_score(validation_y, predicted)
        
        st.markdown(model_name + ": Predicted VS Actual")
        st.markdown("**Red** : Predicted result")
        st.markdown("**Blue** : Actual value")
        ax.plot(validation_y.index, predicted,'r', label='Predict')
        ax.plot(validation_y.index, validation_y.values,'b', label='Actual')
        st.pyplot(fig)
        st.write('RMSE: ', RSME_score)
        st.write('R2 score: ', R2_score)
    
    


        
with model_training:
    st.header('Time to train the model!')
    
    st.subheader('Decision Tree Regressor')
    rs = st.selectbox("Choose the random state for the model.", options = [0,1,2,42,123,2022], index = 3)
    dt = DecisionTreeRegressor(random_state=rs)
    benchmark_dt=dt.fit(X_train, y_train)
    validate_result(benchmark_dt, 'Decision Tree Regression')

    st.subheader('Support Vector Regressor (SVR)')
    svr_lin = SVR(kernel='linear')
    linear_svr_clf_feat = svr_lin.fit(X_train,y_train)
    validate_result(linear_svr_clf_feat,'Linear SVR All Feat')


    st.subheader('Support Vector Regressor (SVR) with GridSearchCV')
    linear_svr_parameters = {
        'C':[0.5, 1.0, 10.0, 50.0],
        'epsilon':[0, 0.1, 0.5, 0.7, 0.9],
    }
    lsvr_grid_search_feat = GridSearchCV(estimator=linear_svr_clf_feat,
                               param_grid=linear_svr_parameters,
                               cv=ts_split,
    )
    lsvr_grid_search_feat.fit(X_train, y_train)
    validate_result(lsvr_grid_search_feat,'Linear SVR GS All Feat')
    
    st.subheader("Random Forest Regressor")
    rf_cl = RandomForestRegressor(n_estimators=50, random_state=0)
    random_forest_clf_feat = rf_cl.fit(X_train,y_train)
    validate_result(random_forest_clf_feat,'Random Forest with All feat')
    
    st.subheader("Random Forest Regressor with Hyper Parameter Tuning")
    random_forest_parameters = {
        'n_estimators':[10,15,20, 50, 100],
        'max_features':['auto','sqrt','log2'],
        'max_depth':[2, 3, 5, 7,10],
    }
    grid_search_RF_feat = GridSearchCV(estimator=random_forest_clf_feat,
                               param_grid=random_forest_parameters,
                               cv=ts_split,
    )
    grid_search_RF_feat.fit(X_train, y_train)
    st.write(grid_search_RF_feat.best_params_)
    validate_result(grid_search_RF_feat,'RandomForest GS') 
    
    st.subheader("Lasso and Ridge")
    from sklearn.linear_model import LassoCV
    from sklearn.linear_model import RidgeCV
    lasso_clf = LassoCV(n_alphas=1000, max_iter=3000, random_state=0)
    ridge_clf = RidgeCV(gcv_mode='auto')
    lasso_clf_feat = lasso_clf.fit(X_train,y_train)
    validate_result(lasso_clf_feat,'LassoCV')
    ridge_clf_feat = ridge_clf.fit(X_train,y_train)
    validate_result(ridge_clf_feat,'RidgeCV')
    
    st.subheader("Bayesian Ridge")
    from sklearn import linear_model
    bay = linear_model.BayesianRidge()
    bay_feat = bay.fit(X_train,y_train)
    validate_result(bay_feat,'Bayesian')

    st.subheader("Gradient Boosting Regressor")
    from sklearn.ensemble import GradientBoostingRegressor
    regr =GradientBoostingRegressor(n_estimators=70, learning_rate=0.1,max_depth=4, random_state=0, loss='ls')
    GB_feat = regr.fit(X_train,y_train)
    validate_result(GB_feat,'NB')
    
    st.subheader("Stochastic Gradient Descent (SGD)")
    from sklearn.linear_model import SGDRegressor
    sgd =SGDRegressor(max_iter=1000, tol=1e-3,loss='squared_epsilon_insensitive',penalty='l1',alpha=0.1)
    sgd_feat = sgd.fit(X_train,y_train)
    validate_result(sgd_feat,'SGD')
    
    
with features:
    st.header('The features I created')
