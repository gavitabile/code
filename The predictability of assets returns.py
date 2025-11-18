import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import skew, kurtosis
import plotly.graph_objs as go
from arch import arch_model
from statsmodels.tsa.ar_model import AutoReg
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import seaborn as sns


#Import data from Excel
filepath = r"C:\Users\gabri\Old Laptop\International Finance\Graf\Empirical Finance\The predictability of asset returns\SP500.xlsx"
SP500 = pd.read_excel(filepath, sheet_name="Sheet1",skiprows=1, index_col=0).reset_index(drop=True)
SP500.head()

# check for null values 
SP500.isnull().sum()

SP500.set_index('Date', inplace=True)

# shape of the dataframe
SP500.shape

SP500['Close Price'].plot(figsize = (15, 8), fontsize = 12)
plt.xlabel('Date', fontsize = 15 )
plt.title('S&P500', fontsize = 20)
plt.legend()
plt.grid()
plt.show()


TBill_monthly = 0.003

Tbill_daily = TBill_monthly * 12 / 250

# Calculate log daily returns
SP500['Returns'] = np.log(SP500['Close Price'] / SP500['Close Price'].shift(1))

SP500['Returns'].plot(figsize = (15, 8), fontsize = 12)
plt.xlabel('Date', fontsize = 15 )
plt.title('S&P500 Returns', fontsize = 20)
plt.legend()
plt.grid()
plt.show()

sns.histplot(SP500['Returns'], kde=True)
plt.xlabel('Date', fontsize = 15 )
plt.title('Distribution of Returns', fontsize = 20)
plt.legend()
plt.grid()
plt.show()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Plot Close Price
ax1.plot(SP500.index, SP500['Close Price'], label='Close Price', color='orange')
ax1.set_title('Close Price')
ax1.legend()

# Plot Forecasted Returns
ax2.plot(SP500.index, SP500['Returns'], label='Returns', color='blue')
ax2.set_title('Returns')
ax2.legend()

plt.show()

print('Skewness:', skew(SP500['Returns']))
print('Kurtosis:', kurtosis(SP500['Returns']))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sm.graphics.tsa.plot_acf(SP500['Returns'], lags=40, ax=axes[0])
sm.graphics.tsa.plot_pacf(SP500['Returns'], lags=40, ax=axes[1])
plt.show()

SP500['T-Bill_Return'] = Tbill_daily

# Drop any NaN values in the Returns column
SP500.dropna(inplace=True)

SP500.head()

trading_days_per_year = 250

# Calculate statistics and collect in a dictionary
statistics = {
    "Average Daily Return (%)": SP500['Returns'].mean() * 100,
    "Average Yearly Return (%)": SP500['Returns'].mean() * trading_days_per_year * 100,
    "Total Return (%)": (np.exp(SP500['Returns'].sum())-1) * 100,
    "Daily Standard Deviation": SP500['Returns'].std(),
    "Annual Standard Deviation": SP500['Returns'].std() * np.sqrt(trading_days_per_year),
    "Skewness": skew(SP500['Returns']),
    "Kurtosis": kurtosis(SP500['Returns']),
    "Maximum Return": SP500['Returns'].max(),
    "Minimum Return": SP500['Returns'].min(),
    "> 0%": (SP500['Returns'] > 0).sum(),
    "< 0%": (SP500['Returns'] < 0).sum(),
    "> 0.1%": (SP500['Returns'] > 0.001).sum(),
    "< -0.1%": (SP500['Returns'] < -0.001).sum(),
    "> 1%": (SP500['Returns'] > 0.01).sum(),
    "< -1%": (SP500['Returns'] < -0.01).sum(),
    "> 10%": (SP500['Returns'] > 0.1).sum(),
    "< -10%": (SP500['Returns'] < -0.1).sum()
}

# Display statistics in a table
print(tabulate(statistics.items(), headers=["Statistic", "Value"]))

#%% TA

Technical_Analysis = pd.DataFrame(SP500["Close Price"])

# Create 1 days simple moving average column
Technical_Analysis['5_SMA'] = Technical_Analysis['Close Price'].rolling(window = 5, min_periods = 1).mean()#.shift(1)

# Create 50 days simple moving average column
Technical_Analysis['150_SMA'] = Technical_Analysis['Close Price'].rolling(window = 150, min_periods = 1).mean()#.shift(1)

# create a new column 'Signal' such that if 1-day SMA is greater than 50-day SMA then set Signal as 1 else 0.
Technical_Analysis['Sign'] = 1.0  
Technical_Analysis['Sign'] = np.where(Technical_Analysis['5_SMA'] >= Technical_Analysis['150_SMA'], 1.0, 0.0) 

# create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
Technical_Analysis['Pos'] = Technical_Analysis['Sign'].diff()

Technical_Analysis['Signal'] = "Hold"  
Technical_Analysis['Signal'] = np.where(Technical_Analysis['Pos'] == 1.0, "Buy", np.where(Technical_Analysis['Pos'] == -1.0,"Sell", "Hold"))

Technical_Analysis['Position'] = "In"  
Technical_Analysis['Position'] = np.where(Technical_Analysis['Sign'].shift(1) == 0.0, "Out", "In") 

Technical_Analysis['Returns'] = 0 
Technical_Analysis['Returns'] = np.where(Technical_Analysis['Position'] == "In", SP500['Returns'], SP500['T-Bill_Return']) 

Technical_Analysis['Excess of Returns'] = Technical_Analysis["Returns"] - SP500["Returns"]

Technical_Analysis['Returns'].plot(figsize = (15, 8), fontsize = 12)
plt.xlabel('Date', fontsize = 15 )
plt.title('Returns of Technical Analysis trading strategy', fontsize = 20)
plt.legend()
plt.grid()
plt.show()

sns.histplot(Technical_Analysis['Returns'], kde=True)
plt.xlabel('Date', fontsize = 15 )
plt.title('Distribution of Returns of TA trading strategy', fontsize = 20)
plt.legend()
plt.grid()
plt.show()

print('Skewness:', skew(Technical_Analysis['Returns']))
print('Kurtosis:', kurtosis(Technical_Analysis['Returns']))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sm.graphics.tsa.plot_acf(Technical_Analysis['Returns'], lags=40, ax=axes[0])
sm.graphics.tsa.plot_pacf(Technical_Analysis['Returns'], lags=40, ax=axes[1])
plt.show()

statistics_Tech_Anal = {
    "Average Daily Return (%)": Technical_Analysis['Returns'].mean() * 100,
    "Average Yearly Return (%)": Technical_Analysis['Returns'].mean() * trading_days_per_year * 100,
    "Total Return (%)": (np.exp(Technical_Analysis['Returns'].sum())-1) * 100,
    "Daily Standard Deviation": Technical_Analysis['Returns'].std(),
    "Annual Standard Deviation": Technical_Analysis['Returns'].std() * np.sqrt(trading_days_per_year),
    "Skewness": skew(Technical_Analysis['Returns']),
    "Kurtosis": kurtosis(Technical_Analysis['Returns']),
    "Maximum Return": Technical_Analysis['Returns'].max(),
    "Minimum Return": Technical_Analysis['Returns'].min(),
    "> 0%": (Technical_Analysis['Returns'] > 0).sum(),
    "< 0%": (Technical_Analysis['Returns'] < 0).sum(),
    "> 0.1%": (Technical_Analysis['Returns'] > 0.001).sum(),
    "< -0.1%": (Technical_Analysis['Returns'] < -0.001).sum(),
    "> 1%": (Technical_Analysis['Returns'] > 0.01).sum(),
    "< -1%": (Technical_Analysis['Returns'] < -0.01).sum(),
    "> 10%": (Technical_Analysis['Returns'] > 0.1).sum(),
    "< -10%": (Technical_Analysis['Returns'] < -0.1).sum(),
    "Signals" : Technical_Analysis['Signal'].value_counts().get('Buy', 0) + Technical_Analysis['Signal'].value_counts().get('Sell', 0),
    "Days In" : Technical_Analysis['Position'].value_counts().get('In', 0),
    "Days Out" : Technical_Analysis['Position'].value_counts().get('Out', 0),
    "Average Excess Return (%)": Technical_Analysis['Excess of Returns'].mean() * 100,
    "Average Yearly Excess Return (%)": Technical_Analysis['Excess of Returns'].mean() * trading_days_per_year * 100,
    "Total Excess of Return (%)": (np.exp(Technical_Analysis['Excess of Returns'].sum())-1) * 100
}




# display the dataframe
Technical_Analysis.head()

plt.figure(figsize = (20,10))
plt.tick_params(axis = 'both', labelsize = 14)
# plot close price, short-term and long-term moving averages 
SP500['Close Price'].plot(color = 'k', lw = 1, label = 'Close Price')  
Technical_Analysis['5_SMA'].plot(color = 'b', lw = 1, label = '5-day SMA') 
Technical_Analysis['150_SMA'].plot(color = 'r', lw = 1, label = '150-day SMA') 

# plot 'buy' signals
plt.plot(Technical_Analysis[Technical_Analysis['Pos'] == 1].index, 
         Technical_Analysis['150_SMA'][Technical_Analysis['Pos'] == 1], 
         '^', markersize = 5, color = 'g', label = 'buy')

# plot 'sell' signals
plt.plot(Technical_Analysis[Technical_Analysis['Pos'] == -1].index, 
         Technical_Analysis['150_SMA'][Technical_Analysis['Pos'] == -1], 
         'v', markersize = 5, color = 'r', label = 'sell')

plt.ylabel('Price', fontsize = 15 )
plt.xlabel('Date', fontsize = 15 )
plt.title('S&P500 - SMA Crossover chart', fontsize = 20)
plt.legend()
plt.grid()
plt.show()






# Create a figure with Plotly
fig = go.Figure()

# Plot close price
fig.add_trace(go.Scatter(x=SP500.index, y=SP500['Close Price'], mode='lines', name='Close Price', line=dict(color='black', width=1)))

# Plot moving averages
fig.add_trace(go.Scatter(x=SP500.index, y=Technical_Analysis['5_SMA'], mode='lines', name='5-day SMA', line=dict(color='blue', width=1)))
fig.add_trace(go.Scatter(x=SP500.index, y=Technical_Analysis['150_SMA'], mode='lines', name='150-day SMA', line=dict(color='red', width=1)))

# Plot buy signals
fig.add_trace(go.Scatter(x=Technical_Analysis[Technical_Analysis['Pos'] == 1].index, 
                         y=Technical_Analysis['150_SMA'][Technical_Analysis['Pos'] == 1], 
                         mode='markers', 
                         name='Buy', 
                         marker=dict(symbol='triangle-up', size=10, color='green')))

# Plot sell signals
fig.add_trace(go.Scatter(x=Technical_Analysis[Technical_Analysis['Pos'] == -1].index, 
                         y=Technical_Analysis['150_SMA'][Technical_Analysis['Pos'] == -1], 
                         mode='markers', 
                         name='Sell', 
                         marker=dict(symbol='triangle-down', size=10, color='red')))

# Update layout
fig.update_layout(title='S&P500 - SMA Crossover Chart',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  legend=dict(x=0, y=1, traceorder='normal'),
                  width=1200, height=600)

# Show plot
fig.show()


fig.write_html(r"C:\Users\Teen\Documents\International Finance\Empirical Finance\\Tech_Anal.html")


#%%AR-GARCH

filepath = r"C:\Users\Teen\Documents\International Finance\Empirical Finance\SP500.xlsx"
AR_SP500 = pd.read_excel(filepath, sheet_name="SP500", skiprows=1, index_col=0).reset_index(drop=True)

# Forward fill missing 'Close Price' values
AR_SP500['Close Price'].ffill(inplace=True)

# Calculate log daily returns
AR_SP500['Returns'] = np.log(AR_SP500['Close Price'] / AR_SP500['Close Price'].shift(1))

# Drop any remaining NaN values
AR_SP500.dropna(inplace=True)

#Step 3: Stationarity Testing
#Use the Augmented Dickey-Fuller (ADF) test to check for stationarity.

result = adfuller(AR_SP500['Returns'])


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sm.graphics.tsa.plot_acf(SP500['Returns'], lags=40, ax=axes[0])
sm.graphics.tsa.plot_pacf(SP500['Returns'], lags=40, ax=axes[1])
plt.show()

print('ADF Statistic:', result[0])
print('p-value:', result[1])
for key, value in result[4].items():
    print('Critical Value (%s): %0.3f' % (key, value))

#Interpretation:

#ADF Statistic: A more negative value indicates stronger evidence against the null hypothesis (i.e., the series is stationary).
#p-value: A p-value less than 0.05 typically indicates the series is stationary.
#Critical Values: If the ADF Statistic is less than the critical value, reject the null hypothesis of non-stationarity.


#Step 4: Autocorrelation and Partial Autocorrelation Analysis
#Examine the autocorrelation (ACF) and partial autocorrelation (PACF) plots.

# ACF and PACF plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sm.graphics.tsa.plot_acf(AR_SP500['Returns'], lags=40, ax=axes[0])
sm.graphics.tsa.plot_pacf(AR_SP500['Returns'], lags=40, ax=axes[1])
plt.show()

#Interpretation:

#ACF Plot: Shows the correlation of the time series with its own lagged values.
#PACF Plot: Shows the correlation of the time series with its own lagged values, controlling for the values of shorter lags.



# Histogram and density plot
sns.histplot(AR_SP500['Returns'], kde=True)
plt.show()

# QQ-plot
sm.qqplot(AR_SP500['Returns'], line='s')
plt.show()

# Skewness and Kurtosis
print('Skewness:', skew(AR_SP500['Returns']))
print('Kurtosis:', kurtosis(AR_SP500['Returns']))

#Interpretation:

#Histogram and Density Plot: Visual check for normality.
#QQ-Plot: Points should lie on the line if the returns are normally distributed.
#Skewness: Measure of asymmetry. A value of 0 means symmetric.
#Kurtosis: Measure of tailedness. A value of 3 indicates normal distribution.


# Split the dataset into training and testing sets
split_ratio = 0.6
split_index = int(split_ratio * len(AR_SP500))
train_data = AR_SP500.iloc[:split_index]
test_data = AR_SP500.iloc[split_index:].reset_index(drop=True)

# Prepare the DataFrame for storing forecast and signal
AR_GARCH = pd.DataFrame({
    'Date': test_data['Date'].values,
    'Close Price': test_data['Close Price'].values,
    'Forecast': np.nan,
    'Volatility': np.nan
})

# Initialize lists to store results
forecasted_returns = []
forecasted_volatilities = []

# Define the window size and lag
window_size = 150

# Iterate over the out-of-sample dataset with a rolling window
for i in range(len(test_data)):
    # Select the rolling window subset
    if i + window_size < len(train_data):
        window_data = train_data['Returns'].iloc[i: i + window_size]
    else:
        window_data = pd.concat([
            train_data['Returns'].iloc[-window_size:],
            pd.Series(forecasted_returns[-(window_size - (len(train_data) - i)):])
        ])
    
    # Fit the AR(1)-GARCH(1,1) model
    model = arch_model(window_data, vol='Garch', p=1, q=1, mean='AR', lags=1)
    model_fit = model.fit(disp='off')
    
    # Forecast next period's return and volatility
    forecast = model_fit.forecast(horizon=1)
    mean_forecast = forecast.mean.iloc[-1].values[0]
    vol_forecast = np.sqrt(forecast.variance.iloc[-1].values[0])
    
    # Handle extreme outliers in forecasted returns
    if mean_forecast > (1.5 * np.std(window_data)):
        mean_forecast = np.median(window_data)
    
    # Store the forecasted returns and volatilities
    forecasted_returns.append(mean_forecast)
    forecasted_volatilities.append(vol_forecast)
    
    # Update the DataFrame
    AR_GARCH.at[i, 'Forecast'] = mean_forecast
    AR_GARCH.at[i, 'Volatility'] = vol_forecast

# Ensure the DataFrame index is aligned with the test data
AR_GARCH.set_index('Date', inplace=True)
test_data.set_index('Date', inplace=True)

d = 0  # Threshold for buy/sell signals

AR_GARCH['Sign'] = 1.0  
AR_GARCH['Sign'] = np.where(AR_GARCH['Forecast'] > d, 1.0, 0.0) 

# create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
AR_GARCH['Pos'] = AR_GARCH['Sign'].diff()

AR_GARCH['Signal'] = "Hold"  
AR_GARCH['Signal'] = np.where(AR_GARCH['Pos'] == 1.0, "Buy", np.where(AR_GARCH['Pos'] == -1.0,"Sell", "Hold"))

AR_GARCH['Position'] = "In"  
AR_GARCH['Position'] = np.where(AR_GARCH['Sign'] == 0.0, "Out", "In")

AR_GARCH.dropna(inplace=True)

AR_GARCH['Returns'] = 0 
AR_GARCH['Returns'] = np.where(AR_GARCH['Position'] == "In", SP500['Returns'], SP500['T-Bill_Return']) 

AR_GARCH['Excess of Returns'] = AR_GARCH["Returns"] - SP500["Returns"]

AR_GARCH['Returns'].plot(figsize = (15, 8), fontsize = 12)
plt.xlabel('Date', fontsize = 15 )
plt.title('Returns of Time Series forecast strategy', fontsize = 20)
plt.legend()
plt.grid()
plt.show()

sns.histplot(AR_GARCH['Returns'], kde=True)
plt.xlabel('Date', fontsize = 15 )
plt.title('Distribution of Returns of TS forecast strategy', fontsize = 20)
plt.legend()
plt.grid()
plt.show()

print('Skewness:', skew(AR_GARCH['Returns']))
print('Kurtosis:', kurtosis(AR_GARCH['Returns']))


statistics_AR_GARCH = {
    "Average Daily Return (%)": AR_GARCH['Returns'].mean() * 100,
    "Average Yearly Return (%)": AR_GARCH['Returns'].mean() * trading_days_per_year * 100,
    "Total Return (%)": (np.exp(AR_GARCH['Returns'].sum())-1) * 100,
    "Daily Standard Deviation": AR_GARCH['Returns'].std(),
    "Annual Standard Deviation": AR_GARCH['Returns'].std() * np.sqrt(trading_days_per_year),
    "Skewness": skew(AR_GARCH['Returns']),
    "Kurtosis": kurtosis(AR_GARCH['Returns']),
    "Maximum Return": AR_GARCH['Returns'].max(),
    "Minimum Return": AR_GARCH['Returns'].min(),
    "> 0%": (AR_GARCH['Returns'] > 0).sum(),
    "< 0%": (AR_GARCH['Returns'] < 0).sum(),
    "> 0.1%": (AR_GARCH['Returns'] > 0.001).sum(),
    "< -0.1%": (AR_GARCH['Returns'] < -0.001).sum(),
    "> 1%": (AR_GARCH['Returns'] > 0.01).sum(),
    "< -1%": (AR_GARCH['Returns'] < -0.01).sum(),
    "> 10%": (AR_GARCH['Returns'] > 0.1).sum(),
    "< -10%": (AR_GARCH['Returns'] < -0.1).sum(),
    "Signals" : AR_GARCH['Signal'].value_counts().get('Buy', 0) + AR_GARCH['Signal'].value_counts().get('Sell', 0),
    "Days In" : AR_GARCH['Position'].value_counts().get('In', 0),
    "Days Out" : AR_GARCH['Position'].value_counts().get('Out', 0),
    "Total Excess of Return (%)": (np.exp(AR_GARCH['Excess of Returns'].sum())-1) * 100
}








#%%







fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Plot Close Price with buy and sell signals
ax1.plot(test_data.index, test_data['Close Price'], label='Close Price')
ax1.plot(AR_GARCH[AR_GARCH['Signal'] == "Buy"].index, 
         test_data['Close Price'][AR_GARCH['Signal'] == "Buy"], 
         '^', markersize=5, color='g', label='Buy')
ax1.plot(AR_GARCH[AR_GARCH['Signal'] == "Sell"].index, 
         test_data['Close Price'][AR_GARCH['Signal'] == "Sell"], 
         'v', markersize=5, color='r', label='Sell')
ax1.set_title('Close Price with Buy and Sell Signals')
ax1.legend()

# Plot Returns
ax2.plot(test_data.index, test_data['Returns'], label='Returns', color='orange')
ax2.set_title('Returns')
ax2.legend()

plt.show()







fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Plot Close Price
ax1.plot(SP500.index, SP500['Close Price'], label='Close Price')
ax1.set_title('Close Price')
ax1.legend()

# Plot Forecasted Returns
ax2.plot(SP500.index, SP500['Returns'], label='Returns', color='orange')
ax2.set_title('Returns')
ax2.legend()

# Add vertical lines for Buy (Sign = 1) and Sell (Sign = 0)
for date in AR_GARCH.index:
    if AR_GARCH.loc[date, 'Pos'] == 1:
        ax1.axvline(x=date, color='green', linestyle='--', label='Buy' if 'Buy' not in ax1.get_legend().get_texts() else "")
        ax2.axvline(x=date, color='green', linestyle='--', label='Buy' if 'Buy' not in ax2.get_legend().get_texts() else "")
    elif AR_GARCH.loc[date, 'Pos'] == 0:
        ax1.axvline(x=date, color='red', linestyle='--', label='Sell' if 'Sell' not in ax1.get_legend().get_texts() else "")
        ax2.axvline(x=date, color='red', linestyle='--', label='Sell' if 'Sell' not in ax2.get_legend().get_texts() else "")

# Ensure the legend does not repeat labels
handles, labels = ax1.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax1.legend(unique_labels.values(), unique_labels.keys())

handles, labels = ax2.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax2.legend(unique_labels.values(), unique_labels.keys())

plt.show()



#%% Combined Strategies

Combined_Strategy = pd.DataFrame(SP500["Close Price"])
#Combined_Strategy['Date'] = SP500['Date']
#Combined_Strategy.set_index('Date', inplace=True)
Combined_Strategy['5_SMA'] = SP500['5_SMA']
Combined_Strategy['150_SMA'] = SP500['150_SMA']
Combined_Strategy["Forecast"] = AR_GARCH["Forecast"]

Combined_Strategy['Sign'] = 1.0  
#Combined_Strategy['Sign'] = np.where((Combined_Strategy["5_SMA"] >= Combined_Strategy['150_SMA']) & (Combined_Strategy["Forecast"] > d), 1.0, 0.0 )
Combined_Strategy['Sign'] = np.where(
    (Combined_Strategy['5_SMA'].shift(1) >= Combined_Strategy['150_SMA'].shift(1)) & (Combined_Strategy['Forecast'] > d), 
    1.0, 
    np.where(
        (Combined_Strategy['5_SMA'].shift(1) < Combined_Strategy['150_SMA'].shift(1)) & (Combined_Strategy['Forecast'] < d),
        0.0,
        np.nan  # Temporary placeholder for the 'otherwise' condition
    )
)
# Fill the 'otherwise' condition with the previous day's value
Combined_Strategy['Sign'] = Combined_Strategy['Sign'].ffill()


# create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
Combined_Strategy['Pos'] = Combined_Strategy['Sign'].diff()

Combined_Strategy['Signal'] = "Hold"  
Combined_Strategy['Signal'] = np.where(Combined_Strategy['Pos'] == 1.0, "Buy", np.where(Combined_Strategy['Pos'] == -1.0,"Sell", "Hold"))

Combined_Strategy['Position'] = "In"  
Combined_Strategy['Position'] = np.where(Combined_Strategy['Sign'] == 0.0, "Out", "In")

Combined_Strategy['Returns'] = 0 
Combined_Strategy['Returns'] = np.where(Combined_Strategy['Position'] == "In", SP500['Returns'], SP500['T-Bill_Return']) 

Combined_Strategy['Excess of Returns'] = Combined_Strategy["Returns"] - SP500["Returns"]


Combined_Strategy['Returns'].plot(figsize = (15, 8), fontsize = 12)
plt.xlabel('Date', fontsize = 15 )
plt.title('Returns of Combined strategies', fontsize = 20)
plt.legend()
plt.grid()
plt.show()

sns.histplot(Combined_Strategy['Returns'], kde=True)
plt.xlabel('Date', fontsize = 15 )
plt.title('Distribution of Returns of combined strategies', fontsize = 20)
plt.legend()
plt.grid()
plt.show()

print('Skewness:', skew(Combined_Strategy['Returns']))
print('Kurtosis:', kurtosis(Combined_Strategy['Returns']))

statistics_Combined_Strategy = {
    "Average Daily Return (%)": Combined_Strategy['Returns'].mean() * 100,
    "Average Yearly Return (%)": Combined_Strategy['Returns'].mean() * trading_days_per_year * 100,
    "Total Return (%)": (np.exp(Combined_Strategy['Returns'].sum())-1) * 100,
    "Daily Standard Deviation": Combined_Strategy['Returns'].std(),
    "Annual Standard Deviation": Combined_Strategy['Returns'].std() * np.sqrt(trading_days_per_year),
    "Skewness": skew(Combined_Strategy['Returns']),
    "Kurtosis": kurtosis(Combined_Strategy['Returns']),
    "Maximum Return": Combined_Strategy['Returns'].max(),
    "Minimum Return": Combined_Strategy['Returns'].min(),
    "> 0%": (Combined_Strategy['Returns'] > 0).sum(),
    "< 0%": (Combined_Strategy['Returns'] < 0).sum(),
    "> 0.1%": (Combined_Strategy['Returns'] > 0.001).sum(),
    "< -0.1%": (Combined_Strategy['Returns'] < -0.001).sum(),
    "> 1%": (Combined_Strategy['Returns'] > 0.01).sum(),
    "< -1%": (Combined_Strategy['Returns'] < -0.01).sum(),
    "> 10%": (Combined_Strategy['Returns'] > 0.1).sum(),
    "< -10%": (Combined_Strategy['Returns'] < -0.1).sum(),
    "Signals" : Combined_Strategy['Signal'].value_counts().get('Buy', 0) + Combined_Strategy['Signal'].value_counts().get('Sell', 0),
    "Days In" : Combined_Strategy['Position'].value_counts().get('In', 0),
    "Days Out" : Combined_Strategy['Position'].value_counts().get('Out', 0),
    "Total Excess of Return (%)": (np.exp(Combined_Strategy['Excess of Returns'].sum())-1) * 100
}

plt.figure(figsize = (20,10))
plt.tick_params(axis = 'both', labelsize = 14)
# plot close price, short-term and long-term moving averages 
Combined_Strategy['Close Price'].plot(color = 'k', lw = 1, label = 'Close Price')  
Combined_Strategy['5_SMA'].plot(color = 'b', lw = 1, label = '5-day SMA') 
Combined_Strategy['150_SMA'].plot(color = 'r', lw = 1, label = '150-day SMA') 

# plot 'buy' signals
plt.plot(Combined_Strategy[Combined_Strategy['Pos'] == 1].index, 
         Combined_Strategy['150_SMA'][Combined_Strategy['Pos'] == 1], 
         '^', markersize = 5, color = 'g',  label = 'buy')

# plot 'sell' signals
plt.plot(Combined_Strategy[Combined_Strategy['Pos'] == -1].index, 
         Combined_Strategy['150_SMA'][Combined_Strategy['Pos'] == -1], 
         'v', markersize = 5, color = 'r',  label = 'sell')

plt.ylabel('Price', fontsize = 15 )
plt.xlabel('Date', fontsize = 15 )
plt.title('S&P500 - Combined Strategy', fontsize = 20)
plt.legend()
plt.grid()
plt.show()






# Create a figure with Plotly
fig = go.Figure()

# Plot close price
fig.add_trace(go.Scatter(x=SP500.index, y=SP500['Close Price'], mode='lines', name='Close Price', line=dict(color='black', width=1)))

# Plot moving averages
fig.add_trace(go.Scatter(x=SP500.index, y=SP500['5_SMA'], mode='lines', name='5-day SMA', line=dict(color='blue', width=1)))
fig.add_trace(go.Scatter(x=SP500.index, y=SP500['150_SMA'], mode='lines', name='150-day SMA', line=dict(color='red', width=1)))

# Plot buy signals
fig.add_trace(go.Scatter(x=Combined_Strategy[Combined_Strategy['Pos'] == 1].index, 
                         y=Combined_Strategy['150_SMA'][Combined_Strategy['Pos'] == 1], 
                         mode='markers', 
                         name='Buy', 
                         marker=dict(symbol='triangle-up', size=10, color='green')))

# Plot sell signals
fig.add_trace(go.Scatter(x=Combined_Strategy[Combined_Strategy['Pos'] == -1].index, 
                         y=Combined_Strategy['150_SMA'][Combined_Strategy['Pos'] == -1], 
                         mode='markers', 
                         name='Sell', 
                         marker=dict(symbol='triangle-down', size=10, color='red')))

# Update layout
fig.update_layout(title='S&P500 - SMA Crossover Chart',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  legend=dict(x=0, y=1, traceorder='normal'),
                  width=1200, height=600)

# Show plot
fig.show()


fig.write_html(r"C:\Users\Teen\Documents\International Finance\Empirical Finance\\Comb_Strat.html")

#%%

sp = np.exp(SP500['Returns'].sum())*10000
ta = np.exp(Technical_Analysis['Returns'].sum())*10000
arg = np.exp(AR_GARCH['Returns'].sum())*10000
combs = np.exp(Combined_Strategy['Returns'].sum())*10000

