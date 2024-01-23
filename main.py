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



f_sales_data = pd.read_csv('./data/flaer_sales.csv', sep=",")
f_sales_data.head()
f_sales_data['Date'] = pd.to_datetime(f_sales_data.Date)
f_sales_data.info()
f_sales_data.describe()



plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

y = f_sales_data['QuantitySold']
y.index.freq = 'D'
y.index = f_sales_data['Date']
y.plot(figsize=(15,6))
plt.show()

f_sales_data.index.freq = 'D'
rcParams['figure.figsize'] = 18, 12
rcParams['figure.autolayout'] = True
decomposition = seasonal_decompose(f_sales_data['QuantitySold'], period=365, model='additive')

decomposition.plot()
plt.show()

#applying exponential smoothing method by differencing succesive rows and stabilizing the mean
f_sales_data = f_sales_data.set_index('Date')
f_sales_data = f_sales_data.rename(columns={'QuantitySold': 'Sales Actual'})

f_sales_data['Sales Actual Adj.'] = f_sales_data['Sales Actual'].diff(365).diff().dropna()
f_sales_data.head(367)

f_sales_data['Sales Actual Adj.'].plot(figsize=(15, 6))
plt.show()


train_sales = f_sales_data[:1461] 
test_sales = f_sales_data[1462:]
train_sales.head()

### 4.&nbsp;Modeling
# 
# Exponential smoothing using the Holt-Winters method is applied to the daily time-series data because it has an upward trend and seasonality features. The forecast aims to provide a medium-term daily view of the sales volumes. Box-Jenkins is used for a longer-term monthly sales forecast.

# #### 4.1 Exponential smoothing using Holt-Winters

# The data is fitted to a model using the Holt-Winters method and plotted to show the actual vs. predicted trend. Holt-Winters is a simple time-series forecasting model that can be used to forecast data with a trend and seasonal component.

fitted_model = ExponentialSmoothing(train_sales['Sales Actual'],trend='add',seasonal='add',seasonal_periods=365).fit()
test_predictions = fitted_model.forecast(len(test_sales))
train_sales['Sales Actual'].plot(legend=True,label='Train Sales', figsize=(15, 6))
test_sales['Sales Actual'].plot(legend=True,label='Test Sales', figsize=(15, 6))
test_predictions.plot(legend=True,label='Forecast Sales')

plt.title('Train, Test and Predicted Test using Holt Winters Exponential Smoothing')


# By inspecting the graph visually, the forecast appears to be a good fit to the actual sales volume in the test set.  
# 
# To further assess the performance of the model, the mean absolute percentage error (MAPE) and root mean square error (RMSE) are used as accuracy metrics.  The `get_mape` function will calculate the MAPE, and the `mean_squared_error` from sklearn is used to calculate the RMSE.
def get_mape(actual, predicted):
    return np.round(np.mean(np.abs((actual-predicted) / actual))*100,2)

hw_mape = get_mape(test_sales['Sales Actual'], test_predictions)
hw_rmse = sqrt(mean_squared_error(test_sales['Sales Actual'], test_predictions))

model_performance = {'Model': ['Holt-Winters'], 
                     'MAPE': hw_mape,
                     'RMSE': hw_rmse}

df_model_results = pd.DataFrame(data=model_performance)

df_model_results

# In[ ]:
flaer_sales_monthly_data = pd.read_csv('./data/flaer_sales_monthly.csv', sep=",")
flaer_sales_monthly_data['SalesMonth'] = pd.to_datetime(flaer_sales_monthly_data.SalesMonth)
flaer_sales_monthly_data = flaer_sales_monthly_data.set_index('SalesMonth')


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
auto_correlation = plot_acf(flaer_sales_monthly_data['Sales Actual'], lags=12)
partial_auto_correlation = plot_pacf(flaer_sales_monthly_data['Sales Actual'], lags=12)


# Analyzing the partial autocorrelation graph shows lag values outside the confidence interval area (shaded blue). If the data was stationary, the lag values would be within the blue shaded area.  
# 
# Applying the Augmented Dickey-Fuller (ADF) test will provide further evidence if the data is non-stationary. ADF tests the null hypothesis that the data is non-stationary. A *p*-value below the acceptable threshold of 0.05 (for the purpose of this example) will reject this null hypothesis and assume the data is stationary.

# ###### 4.2.1.1&nbsp; The Augmented Dickey-Fuller test
dftest = adfuller(flaer_sales_monthly_data['Sales Actual'])
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


# The null hypothesis can be rejected at different critical value thresholds. If the test statistic is less than the 5% threshold but not the 1% threshold, then the null hypothesis is rejected at the 5% threshold. Similarly, if the *p*-value is less than the threshold of 0.05, the null hypothesis will be rejected. However, in this case, the test statistic and *p*-value are greater than the thresholds. Therefore, in this example, the null hypothesis will not be rejected, and you can assume that the data is non-stationary. 

# A transformation will need to be applied to make the data stationary. To do so, the `diff(12)` and `diff()` pandas methods are applied to remove the seasonality (12 months) and trend aspects of the data, respectively.
flaer_sales_monthly_data_diff = flaer_sales_monthly_data['Sales Actual'].diff(12).diff().dropna()





flaer_sales_monthly_data_diff.plot(figsize=(15, 6))
plt.show()


# The distribution of the differencing data does not appear to be seasonal or have a trend. The ADF test is conducted to verify that this is the case.
dftest = adfuller(flaer_sales_monthly_data_diff)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


# The *p*-value is less than the threshold and the differenced data is therefore assumed to be stationary.

# ##### 4.2.2 Splitting the data into training and test sets
# 
# The monthly sales data is split into the training and test data sets using three years of the data for training and the last two years of data for testing.



train_sales_monthly = flaer_sales_monthly_data[:36] 
test_sales_monthly = flaer_sales_monthly_data[37:]


# ##### 4.2.3 Parameter selection
# 
# During the model estimation step, the `auto_arima` method is used to select the best parameters for the SARIMA model. Then, the best values for `p`, `d`, `P`, and `D` are determined by assessing the Akaike information criterion (AIC) score. The combination with the lowest AIC score is selected. As the decomposed trend line is linear and constant over time, the `trend='c'` parameter is passed into the `auto_arima` method.



model = pm.auto_arima(train_sales_monthly['Sales Actual'], d=1, D=1,
                      m=12, seasonal=True, trend='c',
                      start_p=0, start_q=0, test='adf', 
                      stepwise=True, trace=True)


# The lowest AIC score is 562.211. Therefore, the best model parameters are `ARIMA(0,1,0)(0,1,0)[12]`. Using these parameters, a SARIMA model is fitted to the training data. 



model = SARIMAX(train_sales_monthly['Sales Actual'],
                order=(0,1,0),seasonal_order=(0,1,0,12))
results = model.fit(disp=0)
results.summary()


# > **Note:** 
# The `[1] Covariance matrix calculated using the outer product of gradients (complex-step).` warning is a note describing how the covariance matrix was calculated. This can be ignored.

# ##### 4.2.4 Model validation

# For the model validation step, the diagnostics of the forecast results are plotted to analyze the distribution of the residual values. 




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




df_results = pd.concat([test_sales_monthly, forecast_object.predicted_mean], axis=1).dropna()
df_results = df_results.rename(columns={'predicted_mean': 'Sales Forecast'})





bj_mape = get_mape(df_results['Sales Actual'], df_results['Sales Forecast'])
bj_rmse = sqrt(mean_squared_error(df_results['Sales Actual'], df_results['Sales Forecast']))
model_performance = {'Model': ['Box Jenkins'], 
                     'MAPE': bj_mape,
                     'RMSE': bj_rmse}

df_model_results = df_model_results.append(pd.DataFrame(data=model_performance))
df_model_results


