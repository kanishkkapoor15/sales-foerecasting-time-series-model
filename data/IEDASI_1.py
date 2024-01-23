#!/usr/bin/env python
# coding: utf-8

# # IE Predictive Analytics in Business 
# 

# ## Module 4 Unit 2 IDE Practice notebook | Forecasting demand
# 
# The aim of this activity is to provide a framework of the steps required to build a time-series forecasting model to predict demand. The code for each step has been written up for you to execute and is preceded by an explanation to guide you. 
# 
# The two models demonstrated in this activity are the following:
# 
# 1. Exponential smoothing 
# 2. Box-Jenkins 
# 
# Exponential smoothing is used to produce a 12-month daily sales forecast using the Holt-Winters method. The Box-Jenkins approach is used on monthly sales data to produce a longer-term two-year sales forecast. Recall that the Box-Jenkins approach includes all models using an autoregressive (AR) and moving average (MA) process, including the seasonal autoregressive integrated moving average (SARIMA) model.

# ### 1.&nbsp;Import libraries
# 
# In addition to the main libraries used in previous modules, the `statsmodels` library is used in this notebook.
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from pylab import rcParams
from math import sqrt
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import pmdarima as pm

import seaborn as sns

from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings
warnings.filterwarnings('ignore')


# ### 2.&nbsp;Load and explore data
# 
# Once the required libraries have been loaded, the `flaer_sales.csv` data set for T-shirt sales at Flaer is loaded into a pandas data frame. Next, the `describe()`, `info()`, and `head()` methods are used to examine the data. The dates are also converted from a string (text) into a `datetime` object using the pandas `datetime()` method.

# In[ ]:


flaer_sales_data = pd.read_csv('flaer_sales.csv', sep=",")


# In[ ]:


flaer_sales_data.head()


# In[ ]:


flaer_sales_data['Date'] = pd.to_datetime(flaer_sales_data.Date)


# In[ ]:


flaer_sales_data.info()


# In[ ]:


flaer_sales_data.describe()


# > **Pause and reflect:**   
# What information can you gather about the data thus far? How does this data set differ from the other data sets used in this program?

# Next, you can plot the data using `matplotlib` to inspect the data visually. The sales data is recorded daily and the seasonality is based on the annual patterns. As the frequency is daily, it is set to `'D'`. If the data had a 12-month cycle, the frequency of the data set would be set to an annual start (`'AS'`); similarly, monthly data would be set to a monthly start (`'MS'`). 

# In[ ]:


plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# In[ ]:


y = flaer_sales_data['QuantitySold']
y.index.freq = 'D'
y.index = flaer_sales_data['Date']
y.plot(figsize=(15, 6))
plt.show()


# #### 2.1&nbsp;Decompose time-series data
# 
# The `stats_model` `seasonal_decompose` method is used to decompose the observed data into the trend and seasonality (if seasonality applies to the data set). Recall a time series can be seasonally decomposed using two approaches:
# 1. **Additive decomposition:** The time-series data is expressed as a function of the sum of its components.
# 2. **Multiplicative decomposition:** The time-series data is expressed as a function of the product of its components.
# 
# The Flaer sales data is decomposed using the additive approach. 

# In[ ]:


flaer_sales_data.index.freq = 'D'
rcParams['figure.figsize'] = 18, 12
rcParams['figure.autolayout'] = True
decomposition = seasonal_decompose(flaer_sales_data['QuantitySold'], period=365, model='additive')

decomposition.plot()
plt.show()


# The first chart or **observed** chart shows the actual observations per day.
# 
# The **trend** refers to the general direction in which the time series moves over time. In the trend chart, the sales trend has been moving upward linearly over the given time period.
# 
# **Seasonality** refers to the rise and fall of data at consistent frequencies. In the seasonal chart, there is a clear seasonal pattern in an annual cycle with a slight peak in October and a higher peak in September.
# 
# The **residual** refers to what is left over if the trend and seasonality components are removed. As with linear regression, the residuals should be random and centered around zero.
# 
# If the magnitude of the seasonal trend remains constant, the data is considered to be additive.

# ### 3.&nbsp;Prepare the data
# 
# The `QuantitySold` input variable is renamed to `Sales Actual` and differencing is applied to stabilize the mean. 

# In[ ]:


flaer_sales_data = flaer_sales_data.set_index('Date')
flaer_sales_data = flaer_sales_data.rename(columns={'QuantitySold': 'Sales Actual'})


# In[ ]:


flaer_sales_data['Sales Actual Adj.'] = flaer_sales_data['Sales Actual'].diff(365).diff().dropna()
flaer_sales_data.head(367)


# > **Visualize transformed data:**  
# Execute the following code to visualize how seasonality was removed by differencing the data.

# In[ ]:


flaer_sales_data['Sales Actual Adj.'].plot(figsize=(15, 6))
plt.show()


# The sales data is split into training and test data using an 80:20 split. The first 1,461 records are used to train the model and the remaining 364 records are used to test the model.

# In[ ]:


train_sales = flaer_sales_data[:1461] 
test_sales = flaer_sales_data[1462:]


# In[ ]:


train_sales.head()


# ### 4.&nbsp;Modeling
# 
# Exponential smoothing using the Holt-Winters method is applied to the daily time-series data because it has an upward trend and seasonality features. The forecast aims to provide a medium-term daily view of the sales volumes. Box-Jenkins is used for a longer-term monthly sales forecast.

# #### 4.1 Exponential smoothing using Holt-Winters

# The data is fitted to a model using the Holt-Winters method and plotted to show the actual vs. predicted trend. Holt-Winters is a simple time-series forecasting model that can be used to forecast data with a trend and seasonal component.

# In[ ]:


fitted_model = ExponentialSmoothing(train_sales['Sales Actual'],trend='add',seasonal='add',seasonal_periods=365).fit()
test_predictions = fitted_model.forecast(len(test_sales))


# In[ ]:


train_sales['Sales Actual'].plot(legend=True,label='Train Sales', figsize=(15, 6))
test_sales['Sales Actual'].plot(legend=True,label='Test Sales', figsize=(15, 6))
test_predictions.plot(legend=True,label='Forecast Sales')

plt.title('Train, Test and Predicted Test using Holt Winters Exponential Smoothing')


# By inspecting the graph visually, the forecast appears to be a good fit to the actual sales volume in the test set.  
# 
# To further assess the performance of the model, the mean absolute percentage error (MAPE) and root mean square error (RMSE) are used as accuracy metrics.  The `get_mape` function will calculate the MAPE, and the `mean_squared_error` from sklearn is used to calculate the RMSE.

# In[ ]:


def get_mape(actual, predicted):
    return np.round(np.mean(np.abs((actual-predicted) / actual))*100,2)


# In[ ]:


hw_mape = get_mape(test_sales['Sales Actual'], test_predictions)
hw_rmse = sqrt(mean_squared_error(test_sales['Sales Actual'], test_predictions))


# In[ ]:


model_performance = {'Model': ['Holt-Winters'], 
                     'MAPE': hw_mape,
                     'RMSE': hw_rmse}

df_model_results = pd.DataFrame(data=model_performance)
df_model_results


# #### 4.2 Box-Jenkins
# Recall that the Box-Jenkins approach has three steps:  
# 
# 1. Identification
# 2. Estimation
# 3. Validation
# 
# 

# For the Box-Jenkins approach, the Flaer sales data has been aggregated at monthly intervals. The data is loaded and the same steps applied to the daily data set are used to prepare the monthly sales data. The monthly data is also decomposed and plotted to assess seasonality and trend.

# In[ ]:


flaer_sales_monthly_data = pd.read_csv('flaer_sales_monthly.csv', sep=",")
flaer_sales_monthly_data['SalesMonth'] = pd.to_datetime(flaer_sales_monthly_data.SalesMonth)
flaer_sales_monthly_data = flaer_sales_monthly_data.set_index('SalesMonth')


# In[ ]:


flaer_sales_monthly_data.index.freq = 'MS'
flaer_sales_monthly_data = flaer_sales_monthly_data.rename(columns={'QuantitySold': 'Sales Actual'})
rcParams['figure.figsize'] = 18, 12
rcParams['figure.autolayout'] = True
decomposition = seasonal_decompose(flaer_sales_monthly_data['Sales Actual'], period=12, model='additive')

decomposition.plot()
plt.show()


# The monthly sales data appears to have both an upward linear trend and a clear seasonal pattern. This can be confirmed using the three steps of the Box-Jenkins approach.

# ##### 4.2.1 Model identification
# 
# The identification phase determines if the data is stationary. If the data is non-stationary, a transformation is required. The first step is to generate an autocorrelation plot to assess if the data is non-stationary. 

# In[ ]:


auto_correlation = plot_acf(flaer_sales_monthly_data['Sales Actual'], lags=12)
partial_auto_correlation = plot_pacf(flaer_sales_monthly_data['Sales Actual'], lags=12)


# Analyzing the partial autocorrelation graph shows lag values outside the confidence interval area (shaded blue). If the data was stationary, the lag values would be within the blue shaded area.  
# 
# Applying the Augmented Dickey-Fuller (ADF) test will provide further evidence if the data is non-stationary. ADF tests the null hypothesis that the data is non-stationary. A *p*-value below the acceptable threshold of 0.05 (for the purpose of this example) will reject this null hypothesis and assume the data is stationary.

# ###### 4.2.1.1&nbsp; The Augmented Dickey-Fuller test

# In[ ]:


dftest = adfuller(flaer_sales_monthly_data['Sales Actual'])
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


# The null hypothesis can be rejected at different critical value thresholds. If the test statistic is less than the 5% threshold but not the 1% threshold, then the null hypothesis is rejected at the 5% threshold. Similarly, if the *p*-value is less than the threshold of 0.05, the null hypothesis will be rejected. However, in this case, the test statistic and *p*-value are greater than the thresholds. Therefore, in this example, the null hypothesis will not be rejected, and you can assume that the data is non-stationary. 

# A transformation will need to be applied to make the data stationary. To do so, the `diff(12)` and `diff()` pandas methods are applied to remove the seasonality (12 months) and trend aspects of the data, respectively.

# In[ ]:


flaer_sales_monthly_data_diff = flaer_sales_monthly_data['Sales Actual'].diff(12).diff().dropna()


# In[ ]:


flaer_sales_monthly_data_diff.plot(figsize=(15, 6))
plt.show()


# The distribution of the differencing data does not appear to be seasonal or have a trend. The ADF test is conducted to verify that this is the case.

# In[ ]:


dftest = adfuller(flaer_sales_monthly_data_diff)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


# The *p*-value is less than the threshold and the differenced data is therefore assumed to be stationary.

# ##### 4.2.2 Splitting the data into training and test sets
# 
# The monthly sales data is split into the training and test data sets using three years of the data for training and the last two years of data for testing.

# In[ ]:


train_sales_monthly = flaer_sales_monthly_data[:36] 
test_sales_monthly = flaer_sales_monthly_data[37:]


# ##### 4.2.3 Parameter selection
# 
# During the model estimation step, the `auto_arima` method is used to select the best parameters for the SARIMA model. Then, the best values for `p`, `d`, `P`, and `D` are determined by assessing the Akaike information criterion (AIC) score. The combination with the lowest AIC score is selected. As the decomposed trend line is linear and constant over time, the `trend='c'` parameter is passed into the `auto_arima` method.

# In[ ]:


model = pm.auto_arima(train_sales_monthly['Sales Actual'], d=1, D=1,
                      m=12, seasonal=True, trend='c',
                      start_p=0, start_q=0, test='adf', 
                      stepwise=True, trace=True)


# The lowest AIC score is 562.211. Therefore, the best model parameters are `ARIMA(0,1,0)(0,1,0)[12]`. Using these parameters, a SARIMA model is fitted to the training data. 

# In[ ]:


model = SARIMAX(train_sales_monthly['Sales Actual'],
                order=(0,1,0),seasonal_order=(0,1,0,12))
results = model.fit(disp=0)
results.summary()


# > **Note:** 
# The `[1] Covariance matrix calculated using the outer product of gradients (complex-step).` warning is a note describing how the covariance matrix was calculated. This can be ignored.

# ##### 4.2.4 Model validation

# For the model validation step, the diagnostics of the forecast results are plotted to analyze the distribution of the residual values. 

# In[ ]:


results.plot_diagnostics(figsize=(16, 8))
plt.show()


# The **standardized residual for "S"** plot shows the residuals of the forecast. The distribution represents white noise and features a clear seasonal pattern.
# 
# The **histogram** shows that the residuals are somewhat normally distributed.
# 
# Similarly, the ordered distribution of the residuals is plotted in the **Normal Q-Q** chart and illustrates an upward linear trend.
# 
# The **correlogram** shows that the residuals have a low correlation with the lagged versions.
# 
# Based on these charts, it appears that the residuals follow a somewhat normal distribution with a low correlation to the lagged versions. These results indicate that the model is a good fit.
# 
# 

# ##### 4.2.5 Plot the results
# 
# Plotting the actual sales vs. the forecast values provides a visual of the model's performance.

# In[ ]:


flaer_sales_data_monthly_quantity = flaer_sales_monthly_data.drop('AvgTemperatureMonthCelsius', axis = 1)
forecast_object = results.get_forecast(steps=len(test_sales_monthly))
mean = forecast_object.predicted_mean
conf_int = forecast_object.conf_int()
dates = mean.index

ax = flaer_sales_data_monthly_quantity.plot(label='observed')
mean.plot(ax=ax, label='Sales Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(conf_int.index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Month')
ax.set_ylabel('Sales')
plt.legend()
plt.show()


# Based on this visualization, the model appears to be a good fit.  The `MAPE` and the `RMSE` accuracy metrics are used to assess the model's performance.

# In[ ]:


df_results = pd.concat([test_sales_monthly, forecast_object.predicted_mean], axis=1).dropna()
df_results = df_results.rename(columns={'predicted_mean': 'Sales Forecast'})


# In[ ]:


bj_mape = get_mape(df_results['Sales Actual'], df_results['Sales Forecast'])
bj_rmse = sqrt(mean_squared_error(df_results['Sales Actual'], df_results['Sales Forecast']))
model_performance = {'Model': ['Box Jenkins'], 
                     'MAPE': bj_mape,
                     'RMSE': bj_rmse}

df_model_results = df_model_results.append(pd.DataFrame(data=model_performance))
df_model_results


# The Box-Jenkins model performs better than the Holt-Winters based on the MAPE values.  However, these models used different data over different time horizons and should not be directly compared. The RMSE is higher for the Box-Jenkins model as the data was aggregated into monthly buckets, while the Holt-Winters method used daily data. 

# If you have any questions about the process of forecasting seasonal demand, you can submit them for the live session.
