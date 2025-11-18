# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 18:25:10 2025

@author: gabriele.a
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from cvxopt import matrix, solvers
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import xlsxwriter

def get_prices(ticker_symbols, start, end, price_type):
    
    cal = mcal.get_calendar('XNYS')
    schedule = cal.schedule(start_date=start, end_date=end)
    trading_days = schedule.index

    good_symbols = []
    
    try:
        df = yf.download(ticker_symbols[0], start=start, end=end)[[price_type]].rename(columns={price_type: ticker_symbols[0]})
        df = df.reindex(trading_days).interpolate()
        prices = df
        good_symbols.append(ticker_symbols[0])
    except Exception:
        print(f"Download ERROR: {ticker_symbols[0]}")
        prices = pd.DataFrame(index=trading_days)

    for symbol in ticker_symbols[1:]:
        try:
            p = yf.download(symbol, start=start, end=end)[[price_type]].rename(columns={price_type: symbol})
            p = p.reindex(trading_days).interpolate()
            prices = pd.concat([prices, p], axis=1)
            good_symbols.append(symbol)
        except Exception:
            print(f"Download ERROR: {symbol}")

    bad_symbols = []

    for symbol in prices.columns:
        if pd.isna(prices[symbol].iloc[0]) or pd.isna(prices[symbol].iloc[-1]):
            bad_symbols.append(symbol)
        elif prices[symbol].isna().sum() > 0:
            print(f"{symbol} NAs filled: {prices[symbol].isna().sum()}")
            prices[symbol] = prices[symbol].interpolate()

    if bad_symbols:
        prices.drop(columns=bad_symbols, inplace=True)
        print(f"Removed due to NAs: {bad_symbols}")

    if prices.isna().sum().sum() == 0:
        if (prices == 0).sum().sum() > 0:
            print("Check Zeros!")
    else:
        print("Check NAs and Zeros")
        
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    
    prices.index.name = "Date"

    return prices


SPY = ["SPY"] 
PricesSPY = get_prices(SPY, "1999-12-01", "2007-02-28", "Close")

St2010 = ["SPY", "IAU"]
Prices2010 = get_prices(St2010, "2005-02-04", "2011-01-31", "Close")

St2016 = ["SPY", "IAU", "IWDP.SW", "TAN"]
Prices2016 = get_prices(St2016, "2009-01-08", "2016-10-31", "Close")

Stocks = ["SPY", "IAU", "BTC-USD", "IWDP.SW", "TAN"]
Prices = get_prices(Stocks,"2014-10-08", "2025-05-31", "Close")

SPY_compare = get_prices(SPY, "1999-12-01", "2025-05-31", "Close")

#%%
def get_returns(prices):
    
    index = prices.index[1:]  
    columns = prices.columns
    prices = np.asarray(prices)
    
    length = prices.shape[0]
    number_of_stocks = prices.shape[1]

    returns = np.zeros((length - 1, number_of_stocks))

    for ind in range(number_of_stocks):
        returns[:, ind] = np.diff(np.log(prices[:, ind]))
        
    return pd.DataFrame(returns, index=index, columns=columns)

ReturnsSPY = get_returns(PricesSPY)
Returns2010 = get_returns(Prices2010)
Returns2016 = get_returns(Prices2016)
Returns = get_returns(Prices)
Returns_SPY_compare = get_returns(SPY_compare)

#%%
def plot_correlation(cormatrix, labels):
    
    corr_df = pd.DataFrame(cormatrix, index=labels, columns=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_df,
        annot=True,              
        fmt=".2f",           
        cmap="coolwarm",         
        center=0,                
        square=True,             
        linewidths=0.5,          
        cbar_kws={'label': 'Correlation Coefficient'}  
    )
    
    plt.title("Correlation Matrix Heatmap")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

cormat = Returns.corr().values
labels = Returns.columns.tolist()
plot_correlation(cormat, labels)

#%%
def find_weights_for_mu(mu_target, cov_matrix, mean_returns, min_weight ,max_weight):
    N = len(mean_returns)
    
    # Convert to matrices
    P = matrix(2 * cov_matrix)
    q = matrix(np.zeros(N))
    
    # Equality constraints: sum(weights) = 1, weights @ mean_returns = mu_target
    A = matrix(np.vstack([np.ones(N), mean_returns]), tc='d')
    b = matrix([1.0, mu_target])
    
    # Inequality constraints
    # weights ≥ min_weight 
    # weights ≤ max_weight 
    G = np.vstack([
        -np.identity(N),   # -w ≤ -min
        np.identity(N)     #  w ≤ max
    ])
    h = np.hstack([
        -np.ones(N) * min_weight,
         np.ones(N) * max_weight
    ])

    G = matrix(G)
    h = matrix(h)

    # Solve QP
    sol = solvers.qp(P, q, G, h, A, b)
    weights = np.array(sol['x']).flatten()

    # Normalize just in case
    weights /= weights.sum()

    sol['solution'] = weights
    return sol

def rolling_optimal_weights(returns, window, min_weight, max_weight, tickers=None):

    results = []
    
    # Get month-end rebalancing dates
    rebal_dates = returns.index.to_series().groupby(returns.index.to_period("M")).last().values

    for date in tqdm(rebal_dates, desc="Monthly optimization"):
        if date not in returns.index:
            # Skip dates not in the actual index
            continue
        
        end = returns.index.get_loc(date)
        start = end - window
        if start < 0:
            continue  # Not enough data

        past_returns = returns.iloc[start:end]
        cov = np.cov(past_returns, rowvar=False)
        means = np.mean(past_returns, axis=0)
        mus = np.arange(min(means) + 0.00001, max(means) - 0.00001, 0.00001)

        def solve(mus, min_w, max_w):
            feasible_mus = []
            feasible_sigs = []
            feasible_ws = []
            for mu in mus:
                try:
                    sol = find_weights_for_mu(mu, cov, means, min_w, max_w)
                    if sol['status'] == 'optimal':
                        variance = 2 * sol['primal objective']
                        std_dev = np.sqrt(variance)
                        feasible_mus.append(mu)
                        feasible_sigs.append(std_dev)
                        feasible_ws.append(sol['solution'])
                except:
                    continue
            return feasible_mus, feasible_sigs, feasible_ws

        # First attempt
        feasible_mus, feasible_sigs, feasible_ws = solve(mus, min_weight, max_weight)

        # If failed, retry with relaxed bounds
        if not feasible_mus:
            print(f"Retrying with relaxed bounds at {date}")
            feasible_mus, feasible_sigs, feasible_ws = solve(mus, min_weight - 0.05, max_weight + 0.2)

        if feasible_mus:
            sharpe_ratios = np.array(feasible_mus) / np.array(feasible_sigs)
            best_index = np.argmax(sharpe_ratios)
            best_weights = feasible_ws[best_index]
            results.append((date, best_weights))
        else:
            print(f"⚠️ Still no optimal solution at {date}")
            results.append((date, None))

    # Final check for NoneTypes
    none_count = sum(1 for _, w in results if w is None)
    if none_count > 0:
        print(f"\n⚠️ Warning: {none_count} periods still have no optimal solution.")
        
    if tickers is not None:
        data = [(date, w) for date, w in results if w is not None]
        weights_df = pd.DataFrame(
            data=[w for _, w in data],
            index=[pd.to_datetime(date) for date, _ in data],
            columns=tickers
        )
        weights_df.index.name = "Date"
        return weights_df

    return results

weights_by_period10 = rolling_optimal_weights(Returns2010, 500, 0.15, 0.7, tickers=St2010)
weights_by_period16 = rolling_optimal_weights(Returns2016, 500, 0.1, 0.5, tickers=St2016)
weights_by_period = rolling_optimal_weights(Returns, 500, 0.05, 0.4, tickers=Stocks)

#%%
def portfolio_returns(stock_returns, weights):
    if np.isclose(np.sum(weights), 1.0) and len(weights) == stock_returns.shape[1]:
        num_stocks = stock_returns.shape[1]
        length = stock_returns.shape[0]
        P = np.zeros(length)
        
        for t in range(length):
            for d in range(num_stocks):
                P[t] += weights[d] * np.exp(np.sum(stock_returns[:t+1, d]))
        
        P = np.insert(P, 0, 1.0)  
        return np.diff(np.log(P))
    else:
        print("Error: weights do not match")
        return None

def compute_monthly_spy_returns(daily_spy_returns, monthly_dates):
    monthly_spy_returns = []

    for i in range(1, len(monthly_dates)):
        start = monthly_dates[i - 1]
        end = monthly_dates[i]

        # Select daily returns in the period [start, end)
        daily_returns = daily_spy_returns.loc[(daily_spy_returns.index >= start) & (daily_spy_returns.index < end)]

        # Compute cumulative return: product of (1 + daily return), then subtract 1
        if not daily_returns.empty:
            cumulative_spy = daily_returns.values.flatten().sum()
            monthly_spy_returns.append(cumulative_spy)
        else:
            monthly_spy_returns.append(np.nan)  # in case no data is found

    return pd.Series(monthly_spy_returns, index=monthly_dates[1:], name="Monthly_SPY_Returns")

def periodic_portfolio_returns(returns, weights):

    output_returns = []
    period_dates = []

    weights_dates = weights.index
    returns_dates = returns.index

    for i in range(len(weights_dates) - 1):
        start_date = weights_dates[i]
        end_date = weights_dates[i + 1]

        # Slice returns in this monthly window
        period_returns = returns.loc[(returns_dates >= start_date) & (returns_dates < end_date)]
        w = weights.loc[start_date].values

        # Compute portfolio value path with fixed weights
        if np.isclose(w.sum(), 1.0):
            port_return = portfolio_returns(period_returns.values, w)
            monthly_log_return = np.sum(port_return)  # Cumulative log return of the portfolio in that month
            output_returns.append(monthly_log_return)
            period_dates.append(end_date)
        else:
            print(f"Skipped {start_date} due to invalid weights")

    return pd.Series(output_returns, index=period_dates, name="PortfolioReturns")

SPY_month_dates = ReturnsSPY.index.to_series().groupby(ReturnsSPY.index.to_period("M")).last().values
SPY_compare_month_dates = Returns_SPY_compare.index.to_series().groupby(Returns_SPY_compare.index.to_period("M")).last().values

Monthly_SPY_Returns = compute_monthly_spy_returns(ReturnsSPY, SPY_month_dates)
Monthly_Returns10 = periodic_portfolio_returns(Returns2010, weights_by_period10)
Monthly_Returns16 = periodic_portfolio_returns(Returns2016, weights_by_period16)
Monthly_Returns25 = periodic_portfolio_returns(Returns, weights_by_period)
Monthly_Returns = pd.concat([Monthly_SPY_Returns, Monthly_Returns10, Monthly_Returns16, Monthly_Returns25])
Monthly_SPY_compare = compute_monthly_spy_returns(Returns_SPY_compare, SPY_compare_month_dates)

#%%
filepath = r"C:\Users\gabri\Master Thesis\ECB Facility Deposit.xlsx"
ECB_fd = pd.read_excel(filepath, index_col=0, parse_dates=True)

ECB_daily_returns = np.log(1 + ECB_fd / 365)
ECB_daily_returns.columns = ['Daily rf Returns']

def compute_monthly_rf_returns(daily_rf_returns, monthly_dates):
    monthly_rf_returns = []

    for i in range(len(monthly_dates) - 1):
        start = monthly_dates[i]
        end = monthly_dates[i + 1]

        # Select daily rf returns in the period [start, end)
        daily_returns = daily_rf_returns.loc[(daily_rf_returns.index >= start) & (daily_rf_returns.index < end)]

        # Set negative rates to zero
        non_negative_returns = daily_returns.copy()
        non_negative_returns[non_negative_returns < 0] = 0
        
        # Compute cumulative return: product of (1 + daily return), then subtract 1
        if not non_negative_returns.empty:
            cumulative_rf = non_negative_returns.values.flatten().sum()
            monthly_rf_returns.append(cumulative_rf)
        else:
            monthly_rf_returns.append(np.nan)  # in case no data is found

    return pd.Series(monthly_rf_returns, index=monthly_dates[1:], name="Monthly_rf_Returns")

Dec99 = (Monthly_Returns.index[0].to_period("M") - 1).to_timestamp(how="end")
monthly_dates_rf = pd.DatetimeIndex([Dec99]).append(Monthly_Returns.index)

Monthly_rf_Returns = compute_monthly_rf_returns(ECB_daily_returns, monthly_dates_rf)

Discount_rates = Monthly_rf_Returns.loc[Monthly_Returns.index].squeeze()
Discount_rates[Discount_rates < 0] = 0  # Set negative rates to 0 
Discount_rates.name = 'Discount_rates'

transaction_costs = (pd.concat([
    pd.Series(ReturnsSPY.shape[1], index=Monthly_SPY_Returns.index),
    pd.Series(Returns2010.shape[1], index=Monthly_Returns10.index),
    pd.Series(Returns2016.shape[1], index=Monthly_Returns16.index),
    pd.Series(Returns.shape[1], index=Monthly_Returns25.index),
])).loc[Monthly_Returns.index]
transaction_costs.name = "TransactionCosts"

transaction_costs_SPY_compare = pd.Series(1, index=Monthly_SPY_compare.index)
transaction_costs_0 = pd.Series(0, index=Monthly_Returns.index)

Monthly_Contribution = 500

#%%
compareSPY = np.zeros(len(Monthly_SPY_compare))
compareSPY[0] = Monthly_Contribution * np.exp(Monthly_SPY_compare.iloc[0])

for t in range(1, len(Monthly_SPY_compare)):
    compareSPY[t] = (compareSPY[t-1] + Monthly_Contribution - transaction_costs_SPY_compare[t]) * np.exp(Monthly_SPY_compare.iloc[t])

SPY = pd.DataFrame({
    'SPY': np.exp(Monthly_SPY_compare.cumsum()) / np.exp(Monthly_SPY_compare.cumsum()).iloc[0],
    'compareSPY': compareSPY
}, index=Monthly_SPY_compare.index)

plt.plot(SPY.index, SPY["compareSPY"])
plt.show()

comparePORT = np.zeros(len(Monthly_Returns))
comparePORT[0] = Monthly_Contribution * np.exp(Monthly_Returns.iloc[0])

for t in range(1, len(Monthly_Returns)):
    comparePORT[t] = (comparePORT[t-1] + Monthly_Contribution - transaction_costs[t]) * np.exp(Monthly_Returns.iloc[t])

PORT = pd.DataFrame({
    'optPORT': np.exp(Monthly_Returns.cumsum()) / np.exp(Monthly_Returns.cumsum()).iloc[0],
    'comparePORT': comparePORT
}, index=Monthly_Returns.index)

plt.plot(PORT.index, PORT["comparePORT"])
plt.show()

plt.plot(PORT.index, PORT["optPORT"])
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(SPY.index, SPY["SPY"], color="blue", linewidth=2.5, label="SPY")
plt.plot(PORT["optPORT"], color="black", label="Portfolio")
plt.xlabel('Date', fontsize = 14 )
plt.ylabel('Norm Price', fontsize = 14 )
plt.title('SPY vs Diversified Portfolio', fontsize = 20, fontweight = "bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.text(SPY.index[-1], SPY["SPY"].iloc[-1], f"{SPY['SPY'].iloc[-1]:.1f}", color="blue", fontsize=12)
plt.text(SPY.index[-1], PORT["optPORT"].iloc[-1], f"{PORT['optPORT'].iloc[-1]:.1f}", color="black", fontsize=12)
plt.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#%%
month_Len = len(Monthly_Returns)

# === Dynamic Allocation ===
def SimBalAlloc(T, P0, contribution, TransactionCosts, DiscountRates, mL, ReturnSeries1, ReturnSeries2):
    V1 = np.zeros(T)
    V2 = np.zeros(T)
    P = np.zeros(T)
    w = np.zeros(T-1)
    Returns = np.zeros(T)
    cumulative_investment = np.zeros(T)
    c = np.zeros(T)
    cost_paid = np.zeros(T)
    r1_0 = DiscountRates.iloc[0]
    cumulative_investment[0] = P0 + contribution
    w0 = (1 - cumulative_investment[0] /P0 *np.exp(-(r1_0/21)*(T*21))) / mL
    w0 = min(1, max(0, w0))
    V1[0] = P0 * (1 - w0) * np.exp(ReturnSeries1[0])
    V2[0] = P0 * w0 * np.exp(ReturnSeries2[0])
    P[0] = V1[0] + V2[0]
    Returns[0] = ((1-w0) * ReturnSeries1[0]) + (w0 * ReturnSeries2[0])
    for t in range(T - 1):
        cumulative_investment[t+1] = cumulative_investment[t] + contribution
        r1_t = DiscountRates.iloc[t + 1] if t + 1 < T else DiscountRates.iloc[-1]
        w[t] = (1 - cumulative_investment[t+1] / P[t] * np.exp(-r1_t * (T - t))) / mL
        w[t] = min(1, max(0, w[t]))
        if  w[t] == 0 and w[t - 1] == 0:
            c[t + 1] = 0 
        else:
            c[t + 1] = TransactionCosts[t+1]
        cost_paid[t+1] = cost_paid[t] + c[t+1]
        V1[t+1] = (P[t] + contribution - c[t+1]) * (1 - w[t]) * np.exp(ReturnSeries1[t+1])
        V2[t+1] = (P[t] + contribution - c[t+1]) * w[t] * np.exp(ReturnSeries2[t+1])
        P[t+1] = V1[t+1] + V2[t+1] 
        Returns[t+1] = ((1-w[t]) * ReturnSeries1[t+1]) + (w[t] * ReturnSeries2[t+1])
    return {"w": np.concatenate([[w0], w]), "V1": V1, "V2": V2, "P": P, "Returns": Returns, "Investment": cumulative_investment, "TransactionCostPaid": cost_paid}

Dyn_Hyb_Backtest = SimBalAlloc(month_Len, 500, Monthly_Contribution, transaction_costs, Discount_rates, 0.3, Monthly_rf_Returns, Monthly_Returns)
df_Dyn_Hyb = pd.DataFrame(Dyn_Hyb_Backtest, index=Monthly_rf_Returns.index)

plt.figure(figsize=(12, 7))
plt.fill_between(df_Dyn_Hyb.index, df_Dyn_Hyb["P"], color="lightblue", label="Dynamic Hybrid")
plt.plot(df_Dyn_Hyb["P"], color="navy", linewidth=2.5)
plt.fill_between(df_Dyn_Hyb.index, df_Dyn_Hyb["V1"], color="blue", alpha=0.8, label="Risk-free")
plt.plot(df_Dyn_Hyb["Investment"], color='dimgray', label="Investment", linestyle="--")
plt.plot(PORT["comparePORT"], color="black", label="Diversified Portfolio")
plt.xlabel('Date', fontsize = 14)
plt.ylabel('€', fontsize = 14)
plt.title("Dynamic Hybrid Backtest", fontsize = 20, fontweight = "bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.text(df_Dyn_Hyb.index[-1], df_Dyn_Hyb["P"].iloc[-1], f"{df_Dyn_Hyb["P"].iloc[-1]:,.0f}", color="navy", fontsize=12)
plt.text(df_Dyn_Hyb.index[-1], df_Dyn_Hyb["Investment"].iloc[-1], f"{df_Dyn_Hyb["Investment"].iloc[-1]:,.0f}", color="dimgray", fontsize=12)
plt.text(df_Dyn_Hyb.index[-1], df_Dyn_Hyb["V1"].iloc[-1], f"{df_Dyn_Hyb["V1"].iloc[-1]:,.0f}", color="blue", fontsize=12)
plt.text(df_Dyn_Hyb.index[-1], PORT["comparePORT"].iloc[-1], f"{PORT["comparePORT"].iloc[-1]:,.0f}", color="black", fontsize=12)
plt.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#%%
def SimCPPI(T, P0, contribution, TransactionCosts, m, ReturnSeries1, ReturnSeries2):
    V1 = np.zeros(T)
    V2 = np.zeros(T)
    P = np.zeros(T)
    w = np.zeros(T-1)
    Returns = np.zeros(T)
    cumulative_investment = np.zeros(T)
    c = np.zeros(T)
    cost_paid = np.zeros(T)
    floor = np.zeros(T)
    cushion = np.zeros(T)
    floor[0] = P0  # or initial floor
    cushion[0] = max(P0 - floor[0], 0)
    w0 = min((m * cushion[0]) / P0, 1)
    V1[0] = P0 * (1 - w0) * np.exp(ReturnSeries1[0])
    V2[0] = P0 * w0 * np.exp(ReturnSeries2[0])
    P[0] = V1[0] + V2[0]
    cumulative_investment[0] = P0 + contribution
    Returns[0] = ((1-w0) * ReturnSeries1[0]) + (w0 * ReturnSeries2[0])
    for t in range(T - 1):
        cumulative_investment[t+1] = cumulative_investment[t] + contribution
        floor[t+1] = cumulative_investment[t+1]
        cushion[t] = max(P[t] - floor[t+1], 0)
        w[t] = min((m * cushion[t]) / P[t], 1)
        if  w[t] == 0 and w[t - 1] == 0:
            c[t + 1] = 0 
        else:
            c[t + 1] = TransactionCosts[t+1]
        cost_paid[t+1] = cost_paid[t] + c[t+1]
        V1[t+1] = (P[t] + contribution - c[t+1]) * (1 - w[t]) * np.exp(ReturnSeries1[t+1])
        V2[t+1] = (P[t] + contribution - c[t+1]) * w[t] * np.exp(ReturnSeries2[t+1])
        P[t+1] = V1[t+1] + V2[t+1]
        Returns[t+1] = ((1-w[t]) * ReturnSeries1[t+1]) + (w[t] * ReturnSeries2[t+1])
    return {"w": np.concatenate([[w0], w]), "V1": V1, "V2": V2, "P": P, "Returns": Returns, "Investment": cumulative_investment, "TransactionCostPaid": cost_paid, "Floor": floor, "Cushion": cushion}

CPPI_Backtest = SimCPPI(month_Len, 500, Monthly_Contribution, transaction_costs, 2, Monthly_rf_Returns, Monthly_Returns)
df_cppi = pd.DataFrame(CPPI_Backtest, index=Monthly_rf_Returns.index)

plt.figure(figsize=(12, 7))
plt.fill_between(df_cppi.index, df_cppi["P"], color="lightblue", label="CCPI")
plt.plot(df_cppi["P"], color="navy", linewidth=2.5)
plt.fill_between(df_cppi.index, df_cppi["V1"], color="blue", alpha=0.8, label="Risk-free")
plt.plot(df_cppi["Investment"], color='dimgray', label="Investment", linestyle="--")
plt.plot(PORT["comparePORT"], color="black", label="Diversified Portfolio")
plt.xlabel('Date', fontsize = 14)
plt.ylabel('€', fontsize = 14)
plt.title("CPPI Backtest", fontsize = 20, fontweight = "bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.text(df_cppi.index[-1], df_cppi["P"].iloc[-1], f"{df_cppi["P"].iloc[-1]:,.0f}", color="navy", fontsize=12)
plt.text(df_cppi.index[-1], df_cppi["Investment"].iloc[-1], f"{df_cppi["Investment"].iloc[-1]:,.0f}", color="dimgray", fontsize=12)
plt.text(df_cppi.index[-1], df_cppi["V1"].iloc[-1], f"{df_cppi["V1"].iloc[-1]:,.0f}", color="blue", fontsize=12)
plt.text(df_cppi.index[-1], PORT["comparePORT"].iloc[-1], f"{PORT["comparePORT"].iloc[-1]:,.0f}", color="black", fontsize=12)
plt.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#weights evolution
plt.figure(figsize=(12, 7))
plt.plot(df_Dyn_Hyb["w"], color="steelblue", linewidth=2.5, label="Dynamic Hybrid w")
plt.plot(df_cppi["w"], color="darkorange", linewidth=2.5, label="CPPI w")
plt.xlabel('Date', fontsize = 14)
plt.ylabel('w', fontsize = 14)
plt.title("Weights Evolution", fontsize = 20, fontweight = "bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.text(df_Dyn_Hyb.index[-1], df_Dyn_Hyb["w"].iloc[-1] + 0.01, f"{df_Dyn_Hyb["w"].iloc[-1]:.2f}", color="steelblue", fontsize=12)
plt.text(df_cppi.index[-1], df_cppi["w"].iloc[-1] - 0.02, f"{df_cppi["w"].iloc[-1]:.2f}", color="darkorange", fontsize=12)
plt.legend(loc='lower right', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

np.min(Monthly_Returns)

#%%
#Strategies on SPY

#Dynamic Hybrid
Dyn_Hyb_SPY = SimBalAlloc(month_Len, 500, Monthly_Contribution, transaction_costs_SPY_compare, Discount_rates, 0.3, Monthly_rf_Returns, Monthly_SPY_compare)
df_DH_SPY = pd.DataFrame(Dyn_Hyb_SPY, index=Monthly_rf_Returns.index)

plt.figure(figsize=(12, 7))
plt.fill_between(df_DH_SPY.index, df_DH_SPY["P"], color="lightblue", label="Dynamic Hybrid")
plt.plot(df_DH_SPY["P"], color="navy", linewidth=2.5)
plt.fill_between(df_DH_SPY.index, df_DH_SPY["V1"], color="blue", alpha=0.8, label="Risk-free")
plt.plot(df_DH_SPY["Investment"], color='dimgray', label="Investment", linestyle="--")
plt.plot(SPY["compareSPY"], color="black", label="SPY")
plt.xlabel('Date', fontsize = 14)
plt.ylabel('€', fontsize = 14)
plt.title("Dynamic Hybrid SPY", fontsize = 20, fontweight = "bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.text(df_DH_SPY.index[-1], df_DH_SPY["P"].iloc[-1] - 10000, f"{df_DH_SPY["P"].iloc[-1]:,.0f}", color="navy", fontsize=12)
plt.text(df_DH_SPY.index[-1], df_DH_SPY["Investment"].iloc[-1] + 15000, f"{df_DH_SPY["Investment"].iloc[-1]:,.0f}", color="dimgray", fontsize=12)
plt.text(df_DH_SPY.index[-1], df_DH_SPY["V1"].iloc[-1] - 10000, f"{df_DH_SPY["V1"].iloc[-1]:,.0f}", color="blue", fontsize=12)
plt.text(df_DH_SPY.index[-1], SPY["compareSPY"].iloc[-1], f"{SPY["compareSPY"].iloc[-1]:,.0f}", color="black", fontsize=12)
plt.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#CPPI
CPPI_SPY = SimCPPI(month_Len, 500, Monthly_Contribution, transaction_costs_SPY_compare, 2, Monthly_rf_Returns, Monthly_SPY_compare)
df_cppi_SPY = pd.DataFrame(CPPI_SPY, index=Monthly_rf_Returns.index)

plt.figure(figsize=(12, 7))
plt.fill_between(df_cppi_SPY.index, df_cppi_SPY["P"], color="lightblue", label="CCPI")
plt.plot(df_cppi_SPY["P"], color="navy", linewidth=2.5)
plt.fill_between(df_cppi_SPY.index, df_cppi_SPY["V1"], color="blue", alpha=0.8, label="Risk-free")
plt.plot(df_cppi_SPY["Investment"], color='dimgray', label="Investment", linestyle="--")
plt.plot(SPY["compareSPY"], color="black", label="SPY")
plt.xlabel('Date', fontsize = 14)
plt.ylabel('€', fontsize = 14)
plt.title("CPPI SPY", fontsize = 20, fontweight = "bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.text(df_cppi_SPY.index[-1], df_cppi_SPY["P"].iloc[-1], f"{df_cppi_SPY["P"].iloc[-1]:,.0f}", color="navy", fontsize=12)
plt.text(df_cppi_SPY.index[-1], df_cppi_SPY["Investment"].iloc[-1], f"{df_cppi_SPY["Investment"].iloc[-1]:,.0f}", color="dimgray", fontsize=12)
plt.text(df_cppi_SPY.index[-1], df_cppi_SPY["V1"].iloc[-1], f"{df_cppi_SPY["V1"].iloc[-1]:,.0f}", color="blue", fontsize=12)
plt.text(df_cppi_SPY.index[-1], SPY["compareSPY"].iloc[-1], f"{SPY["compareSPY"].iloc[-1]:,.0f}", color="black", fontsize=12)
plt.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#Weights SPY
plt.figure(figsize=(12, 7))
plt.plot(df_DH_SPY["w"], color="steelblue", linewidth=2.5, label="Dynamic Hybrid w")
plt.plot(df_cppi_SPY["w"], color="darkorange", linewidth=2.5, label="CPPI w")
plt.xlabel('Date', fontsize = 14)
plt.ylabel('w', fontsize = 14)
plt.title("Weights Evolution", fontsize = 20, fontweight = "bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.text(df_DH_SPY.index[-1], df_DH_SPY["w"].iloc[-1], f"{df_DH_SPY["w"].iloc[-1]:.2f}", color="steelblue", fontsize=12)
plt.text(df_cppi_SPY.index[-1], df_cppi_SPY["w"].iloc[-1], f"{df_cppi_SPY["w"].iloc[-1]:.2f}", color="darkorange", fontsize=12)
plt.legend(loc='center right', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#Weights PORT vs SPY Dyn Hyb
plt.figure(figsize=(12, 7))
plt.plot(df_Dyn_Hyb["w"], color="steelblue", linewidth=2.5, label="Diversified Portfolio w")
plt.plot(df_DH_SPY["w"], color="darkorange", linewidth=2.5, label="SPY w")
plt.xlabel('Date', fontsize = 14)
plt.ylabel('w', fontsize = 14)
plt.title("Diversification Effect on Dyn Hyb Weights ", fontsize = 20, fontweight = "bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.text(df_Dyn_Hyb.index[-1], df_Dyn_Hyb["w"].iloc[-1], f"{df_Dyn_Hyb["w"].iloc[-1]:.2f}", color="steelblue", fontsize=12)
plt.text(df_DH_SPY.index[-1], df_DH_SPY["w"].iloc[-1], f"{df_DH_SPY["w"].iloc[-1]:.2f}", color="darkorange", fontsize=12)
plt.legend(loc='center right', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#Weights PORT vs SPY CPPI
plt.figure(figsize=(12, 7))
plt.plot(df_cppi["w"], color="steelblue", linewidth=2.5, label="Diversified Portfolio w")
plt.plot(df_cppi_SPY["w"], color="darkorange", linewidth=2.5, label="SPY w")
plt.xlabel('Date', fontsize = 14)
plt.ylabel('w', fontsize = 14)
plt.title("Diversification Effect on CPPI Weights", fontsize = 20, fontweight = "bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.text(df_cppi.index[-1], df_cppi["w"].iloc[-1], f"{df_cppi["w"].iloc[-1]:.2f}", color="steelblue", fontsize=12)
plt.text(df_cppi_SPY.index[-1], df_cppi_SPY["w"].iloc[-1] - 0.02, f"{df_cppi_SPY["w"].iloc[-1]:.2f}", color="darkorange", fontsize=12)
plt.legend(loc='center right', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#%%
#Strategies with tc = 0

#Dyn Hyb
Dyn_Hyb_tc0 = SimBalAlloc(month_Len, 500, Monthly_Contribution, transaction_costs_0, Discount_rates, 0.3, Monthly_rf_Returns, Monthly_Returns)
df_Dyn_Hyb_tc0 = pd.DataFrame(Dyn_Hyb_tc0, index=Monthly_rf_Returns.index)

plt.figure(figsize=(12, 7))
plt.fill_between(df_Dyn_Hyb_tc0.index, df_Dyn_Hyb_tc0["P"], color="lightblue", label="Dynamic Hybrid")
plt.plot(df_Dyn_Hyb_tc0["P"], color="navy", linewidth=2.5)
plt.fill_between(df_Dyn_Hyb_tc0.index, df_Dyn_Hyb_tc0["V1"], color="blue", alpha=0.8, label="Risk-free")
plt.plot(df_Dyn_Hyb_tc0["Investment"], color='dimgray', label="Investment", linestyle="--")
plt.plot(PORT["comparePORT"], color="black", label="Diversified Portfolio")
plt.xlabel('Date', fontsize = 14)
plt.ylabel('€', fontsize = 14)
plt.title("Dynamic Hybrid TC = 0", fontsize = 20, fontweight = "bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.text(df_Dyn_Hyb_tc0.index[-1], df_Dyn_Hyb_tc0["P"].iloc[-1], f"{df_Dyn_Hyb_tc0["P"].iloc[-1]:,.0f}", color="navy", fontsize=12)
plt.text(df_Dyn_Hyb_tc0.index[-1], df_Dyn_Hyb_tc0["Investment"].iloc[-1], f"{df_Dyn_Hyb_tc0["Investment"].iloc[-1]:,.0f}", color="dimgray", fontsize=12)
plt.text(df_Dyn_Hyb_tc0.index[-1], df_Dyn_Hyb_tc0["V1"].iloc[-1], f"{df_Dyn_Hyb_tc0["V1"].iloc[-1]:,.0f}", color="blue", fontsize=12)
plt.text(df_Dyn_Hyb_tc0.index[-1], PORT["comparePORT"].iloc[-1], f"{PORT["comparePORT"].iloc[-1]:,.0f}", color="black", fontsize=12)
plt.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#CPPI
CPPI_tc0 = SimCPPI(month_Len, 500, Monthly_Contribution, transaction_costs_0, 2, Monthly_rf_Returns, Monthly_Returns)
df_cppi_tc0 = pd.DataFrame(CPPI_tc0, index=Monthly_rf_Returns.index)

plt.figure(figsize=(12, 7))
plt.fill_between(df_cppi_tc0.index, df_cppi_tc0["P"], color="lightblue", label="CCPI")
plt.plot(df_cppi_tc0["P"], color="navy", linewidth=2.5)
plt.fill_between(df_cppi_tc0.index, df_cppi_tc0["V1"], color="blue", alpha=0.8, label="Risk-free")
plt.plot(df_cppi_tc0["Investment"], color='dimgray', label="Investment", linestyle="--")
plt.plot(PORT["comparePORT"], color="black", label="Diversified Portfolio")
plt.xlabel('Date', fontsize = 14)
plt.ylabel('€', fontsize = 14)
plt.title("CPPI TC = 0", fontsize = 20, fontweight = "bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.text(df_cppi_tc0.index[-1], df_cppi_tc0["P"].iloc[-1], f"{df_cppi_tc0["P"].iloc[-1]:,.0f}", color="navy", fontsize=12)
plt.text(df_cppi_tc0.index[-1], df_cppi_tc0["Investment"].iloc[-1], f"{df_cppi_tc0["Investment"].iloc[-1]:,.0f}", color="dimgray", fontsize=12)
plt.text(df_cppi_tc0.index[-1], df_cppi_tc0["V1"].iloc[-1], f"{df_cppi_tc0["V1"].iloc[-1]:,.0f}", color="blue", fontsize=12)
plt.text(df_cppi_tc0.index[-1], PORT["comparePORT"].iloc[-1], f"{PORT["comparePORT"].iloc[-1]:,.0f}", color="black", fontsize=12)
plt.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#Tc vs noTC
plt.figure(figsize=(12, 7))
#plt.plot(df_Dyn_Hyb["w"], color="steelblue", linewidth=2.5, label="Dynamic Hybrid w")
plt.plot(df_cppi["w"], color="darkorange", linewidth=2.5, label="CPPI w")
#plt.plot(df_Dyn_Hyb_tc0["w"], color="darkorange", linestyle="--", label="Dyn Hyb TC0 w")
plt.plot(df_cppi_tc0["w"], color="steelblue", linestyle="--", label="CPPI TC0 w")
plt.xlabel('Date', fontsize = 14)
plt.ylabel('w', fontsize = 14)
plt.title("Weights Evolution", fontsize = 20, fontweight = "bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
#plt.text(df_Dyn_Hyb.index[-1], df_Dyn_Hyb["w"].iloc[-1] + 0.01, f"{df_Dyn_Hyb["w"].iloc[-1]:.2f}", color="steelblue", fontsize=12)
plt.text(df_cppi.index[-1], df_cppi["w"].iloc[-1] - 0.02, f"{df_cppi["w"].iloc[-1]:.2f}", color="darkorange", fontsize=12)
#plt.text(df_Dyn_Hyb_tc0.index[-1], df_Dyn_Hyb_tc0["w"].iloc[-1] + 0.01, f"{df_Dyn_Hyb_tc0["w"].iloc[-1]:.2f}", color="darkorange", fontsize=12)
plt.text(df_cppi_tc0.index[-1], df_cppi_tc0["w"].iloc[-1] - 0.02, f"{df_cppi_tc0["w"].iloc[-1]:.2f}", color="steelblue", fontsize=12)
plt.legend(loc='lower right', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#DH vs CPPI noTC
plt.figure(figsize=(12, 7))
plt.plot(df_Dyn_Hyb_tc0["w"], color="steelblue", linewidth=2.5, label="Dynamic Hybrid w")
plt.plot(df_cppi_tc0["w"], color="darkorange", linewidth=2.5, label="CPPI w")
plt.xlabel('Date', fontsize = 14)
plt.ylabel('w', fontsize = 14)
plt.title("Weights TC = 0", fontsize = 20, fontweight = "bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.text(df_Dyn_Hyb_tc0.index[-1], df_Dyn_Hyb_tc0["w"].iloc[-1] + 0.01, f"{df_Dyn_Hyb_tc0["w"].iloc[-1]:.2f}", color="steelblue", fontsize=12)
plt.text(df_cppi_tc0.index[-1], df_cppi_tc0["w"].iloc[-1] - 0.02, f"{df_cppi_tc0["w"].iloc[-1]:.2f}", color="darkorange", fontsize=12)
plt.legend(loc='lower right', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#Useless
plt.plot(df_Dyn_Hyb["TransactionCostPaid"], label="Dynamic Hybrid")
plt.plot(df_cppi["TransactionCostPaid"], label="CPPI")
plt.title("Costs over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
transaction_costs_opt = 5*month_Len

def value_at_risk(returns, alpha):
    sorted_returns = np.sort(returns)
    index = int(np.ceil((1 - alpha) * len(sorted_returns)))
    return sorted_returns[index - 1]  # zero-based index

def expected_shortfall(returns, alpha):
    sorted_returns = np.sort(returns)
    cutoff = int(np.ceil((1 - alpha) * len(sorted_returns)))
    return np.mean(sorted_returns[:cutoff]) if cutoff > 0 else np.nan

def max_drawdown(returns):
    cumulative = np.exp(returns.cumsum())
    roll_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - roll_max) / roll_max
    return drawdown.min()

def compute_metrics(returns, rf, alpha):
    returns = np.asarray(returns)
    
    mean = np.mean(returns)
    std = np.std(returns)
    downside_std = np.std(returns[returns < 0])
    
    sharpe = (mean - rf) / std if std > 0 else np.nan
    sortino = (mean - rf) / downside_std if downside_std > 0 else np.nan
    mdd = max_drawdown(returns)
    var = value_at_risk(returns, alpha)
    es = expected_shortfall(returns, alpha)
    
    return {
        "Mean": mean,
        "Std": std,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": mdd,
        f"VaR ({alpha:.0%})": var,
        f"ES ({alpha:.0%})": es
    }

rf = np.mean(Monthly_rf_Returns)

def strategy_metrics_dataframe(return_df, rf, alpha):
    metrics_dict = {}
    for col in return_df.columns:
        metrics = compute_metrics(return_df[col].dropna(), rf, alpha)
        metrics_dict[col] = metrics
    return pd.DataFrame(metrics_dict).T

metrics_df = strategy_metrics_dataframe(pd.DataFrame({
    "CPPI": np.array(df_cppi["Returns"]),
    "Dynamic Hybrid": np.array(df_Dyn_Hyb["Returns"]),
    "Optimized": Monthly_Returns,
    "CPPI_SPY": np.array(df_cppi_SPY["Returns"]),
    "Dynamic Hybrid_SPY": np.array(df_DH_SPY["Returns"]),
    "CPPI_tc0": np.array(df_cppi_tc0["Returns"]),
    "Dynamic Hybrid_tc0": np.array(df_Dyn_Hyb_tc0["Returns"]),
}), rf, 0.95)

Terminal_Value = pd.DataFrame({
    "CPPI": [df_cppi["P"].iloc[-1]],
    "Dynamic Hybrid": [df_Dyn_Hyb["P"].iloc[-1]],
    "Optimized": [PORT["comparePORT"].iloc[-1]],
    "CPPI_SPY": [df_cppi_SPY["P"].iloc[-1]],
    "Dynamic Hybrid_SPY": [df_DH_SPY["P"].iloc[-1]],
    "CPPI_tc0": [df_cppi_tc0["P"].iloc[-1]],
    "Dynamic Hybrid_tc0": [df_Dyn_Hyb_tc0["P"].iloc[-1]],
})

Total_Cost = pd.DataFrame({
    "CPPI": [df_cppi["TransactionCostPaid"].iloc[-1]],
    "Dynamic Hybrid": [df_Dyn_Hyb["TransactionCostPaid"].iloc[-1]],
    "Optimized": transaction_costs.sum(),
    "CPPI_SPY": [df_cppi_SPY["TransactionCostPaid"].iloc[-1]],
    "Dynamic Hybrid_SPY": [df_DH_SPY["TransactionCostPaid"].iloc[-1]],
    "CPPI_tc0": [df_cppi_tc0["TransactionCostPaid"].iloc[-1]],
    "Dynamic Hybrid_tc0": [df_Dyn_Hyb_tc0["TransactionCostPaid"].iloc[-1]],
})

metrics_df.insert(0, "Terminal Value (€)", Terminal_Value.iloc[0])
metrics_df["Total Cost (€)"] = Total_Cost.iloc[0]

percentage_cols = ["Mean", "Std", "MaxDrawdown", "VaR (95%)", "ES (95%)"]
for col in percentage_cols:
    metrics_df[col] = (metrics_df[col] * 100).map("{:,.2f}%".format)

metrics_df["Terminal Value (€)"] = metrics_df["Terminal Value (€)"].map("{:,.0f}".format)
metrics_df["Total Cost (€)"] = metrics_df["Total Cost (€)"].map("{:,.0f}".format)

metrics_df["Sharpe"] = metrics_df["Sharpe"].round(3)
metrics_df["Sortino"] = metrics_df["Sortino"].round(3)

TC_FinalValue = (Total_Cost.iloc[0] / Terminal_Value.iloc[0]) * 100

total_contrib = df_Dyn_Hyb["Investment"].iloc[-1]
TC_Investment = (Total_Cost.iloc[0] / total_contrib) * 100

#TC_UnitofReturn = Total_Cost.iloc[0] / (Terminal_Value.iloc[0] - total_contrib)

tc_drag_values = {
    "CPPI": (np.exp(df_cppi_tc0["Returns"].sum()) - np.exp(df_cppi["Returns"].sum())) * 100,
    "Dynamic Hybrid": (np.exp(df_Dyn_Hyb_tc0["Returns"].sum()) - np.exp(df_Dyn_Hyb["Returns"].sum())) * 100,
}

metrics_df["TC % Final Value"] = TC_FinalValue.map("{:,.2f}%".format)
metrics_df["TC % Contribution"] = TC_Investment.map("{:,.2f}%".format)
#metrics_df["TC per Unit Return"] = TC_UnitofReturn.map("{:,.6f}".format)
metrics_df["TC Drag (p.p.)"] = metrics_df.index.map(lambda name: f"{tc_drag_values.get(name, np.nan):.2f}")


with pd.ExcelWriter("strategy_metrics.xlsx", engine="xlsxwriter") as writer:
    metrics_df.to_excel(writer, sheet_name="Metrics", startrow=1, header=False)

    workbook  = writer.book
    worksheet = writer.sheets["Metrics"]

    # Write the column headers manually with bold format
    header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top',
                                         'align': 'center', 'border': 1, 'bg_color': '#D7E4BC'})

    for col_num, value in enumerate(metrics_df.columns.values):
        worksheet.write(0, col_num + 1, value, header_format)

    # Format the index column
    index_format = workbook.add_format({'align': 'left', 'bold': True, 'border': 1})
    for row_num, idx in enumerate(metrics_df.index):
        worksheet.write(row_num + 1, 0, idx, index_format)

    # Optional: auto-fit column widths
    for i, col in enumerate(metrics_df.columns):
        max_len = max(metrics_df[col].astype(str).map(len).max(), len(col)) + 2
        worksheet.set_column(i + 1, i + 1, max_len)

#%%       
FinalValues_DH = np.zeros(1000)
FinalValues_CPPI = np.zeros(1000)
FinalValuesPORT = np.zeros(1000)
FinalValues_DH_SPY = np.zeros(1000)
FinalValues_CPPI_SPY = np.zeros(1000)
FinalValuesSPY = np.zeros(1000)

Mean_port, STD_port = np.mean(Monthly_Returns), np.std(Monthly_Returns)
Mean_spy, STD_spy = np.mean(Monthly_SPY_compare), np.std(Monthly_SPY_compare)

TenY = 12 * 10
TenY_index = pd.date_range(start=Monthly_Returns.index[-1] + pd.offsets.MonthBegin(1), periods=TenY, freq='M')

transaction_costs_sim = (pd.Series(5, index=TenY_index)).loc[TenY_index]
transaction_costs_sim.name = "TransactionCosts"
transaction_costs_spy_sim = (pd.Series(1, index=TenY_index)).loc[TenY_index]
transaction_costs_spy_sim.name = "TransactionCostsSPY"

current_ECB_fd = 0.02  # 2% annual as of June 2025
monthly_rf_sim = np.log(1 + current_ECB_fd) / 12

Discount_rates_sim = (pd.Series(monthly_rf_sim, index=TenY_index)).loc[TenY_index]
Discount_rates_sim.name = "DiscountRates"

RetFI = np.full(TenY, monthly_rf_sim)

for i in range(1000):
    RetPORT_Sim = np.random.normal(Mean_port, STD_port, TenY)
    RetSPY = np.random.normal(Mean_spy, STD_spy, TenY)
    
    DH_Sim = SimBalAlloc(TenY, 500, Monthly_Contribution, transaction_costs_sim, Discount_rates_sim, 0.3, RetFI, RetPORT_Sim)
    CPPI_Sim = SimCPPI(TenY, 500, Monthly_Contribution, transaction_costs_sim, 2, RetFI, RetPORT_Sim)
    
    DH_Sim_spy = SimBalAlloc(TenY, 500, Monthly_Contribution, transaction_costs_spy_sim, Discount_rates_sim, 0.3, RetFI, RetSPY)
    CPPI_Sim_spy = SimCPPI(TenY, 500, Monthly_Contribution, transaction_costs_spy_sim, 2, RetFI, RetSPY)    
    
    FinalValues_DH[i] = DH_Sim["P"][-1]
    FinalValues_CPPI[i] = CPPI_Sim["P"][-1]
    
    FinalValues_DH_SPY[i] = DH_Sim_spy["P"][-1]
    FinalValues_CPPI_SPY[i] = CPPI_Sim_spy["P"][-1]
    
    port_sim = np.zeros(TenY)
    port_sim[0] = 500 * np.exp(RetPORT_Sim[0])
       
    for t in range(1, TenY):
        port_sim[t] = (port_sim[t-1] + Monthly_Contribution - transaction_costs_sim[t]) * np.exp(RetPORT_Sim[t])
    
    FinalValuesPORT[i] = port_sim[-1]
    
    spy_sim = np.zeros(TenY)
    spy_sim[0] = 500 * np.exp(RetSPY[0])
       
    for t in range(1, TenY):
        spy_sim[t] = (spy_sim[t-1] + Monthly_Contribution - transaction_costs_spy_sim[t]) * np.exp(RetSPY[t])
        
    FinalValuesSPY[i] = spy_sim[-1]

tot_contribution = TenY * Monthly_Contribution

success_rate_dh = np.mean(FinalValues_DH >= tot_contribution)
success_rate_cppi = np.mean(FinalValues_CPPI >= tot_contribution)
success_rate_port = np.mean(FinalValuesPORT >= tot_contribution)
success_rate_dh_spy = np.mean(FinalValues_DH_SPY >= tot_contribution)
success_rate_cppi_spy = np.mean(FinalValues_CPPI_SPY >= tot_contribution)
success_rate_spy = np.mean(FinalValuesSPY >= tot_contribution)

plt.figure(figsize=(9.5, 7))
sns.histplot(FinalValues_DH, stat="density", bins=50, color="blue", alpha=0.2, edgecolor=None)
sns.histplot(FinalValues_CPPI, stat="density", bins=50, color="red", alpha=0.2, edgecolor=None)
sns.histplot(FinalValuesPORT, stat="density", bins=50, color="dimgray", alpha=0.2, edgecolor=None)
sns.kdeplot(FinalValues_DH, label="Dynamic Hybrid", color="blue", linewidth=2.5, bw_adjust=0.8, clip=(np.min(FinalValues_DH), np.max(FinalValues_DH)))
sns.kdeplot(FinalValues_CPPI, label="CPPI", color="red", linewidth=2.5, bw_adjust=0.8, clip=(np.min(FinalValues_CPPI), np.max(FinalValues_CPPI)))
sns.kdeplot(FinalValuesPORT, label="Optimized Portfolio", color="dimgray", linewidth=2, bw_adjust=0.8, clip=(np.min(FinalValuesPORT), np.max(FinalValuesPORT)))
plt.axvline(tot_contribution, color="black", linestyle="--", linewidth=1.0, label=f"Total Contributions (€{tot_contribution/1000:.0f}k)", alpha=0.9)
plt.axvline(np.mean(FinalValues_DH), color="blue", linestyle=":", linewidth=1.2, alpha=0.8)
plt.axvline(np.mean(FinalValues_CPPI), color="red", linestyle=":", linewidth=1.2, alpha=0.8)
plt.axvline(np.mean(FinalValuesPORT), color="dimgray", linestyle=":", linewidth=1.2, alpha=0.8)
plt.xlabel("Final Value (€)", fontsize=14)
plt.ylabel("Density (%)", fontsize=14)
plt.title("Distribution of Final Values", fontsize=20, fontweight="bold")
plt.legend(loc='upper right', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x * 100:.3f}%'))
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#SPY Sim
plt.figure(figsize=(9.5, 7))
sns.histplot(FinalValues_DH_SPY, stat="density", bins=50, color="blue", alpha=0.2, edgecolor=None)
sns.histplot(FinalValues_CPPI_SPY, stat="density", bins=50, color="red", alpha=0.2, edgecolor=None)
sns.histplot(FinalValuesSPY, stat="density", bins=50, color="dimgray", alpha=0.2, edgecolor=None)
sns.kdeplot(FinalValues_DH_SPY, label="Dynamic Hybrid SPY", color="blue", linewidth=2.5, bw_adjust=0.8, clip=(np.min(FinalValues_DH_SPY), np.max(FinalValues_DH_SPY)))
sns.kdeplot(FinalValues_CPPI_SPY, label="CPPI SPY", color="red", linewidth=2.5, bw_adjust=0.8, clip=(np.min(FinalValues_CPPI_SPY), np.max(FinalValues_CPPI_SPY)))
sns.kdeplot(FinalValuesSPY, label="SPY", color="dimgray", linewidth=2, bw_adjust=0.8, clip=(np.min(FinalValuesSPY), np.max(FinalValuesSPY)))
plt.axvline(tot_contribution, color="black", linestyle="--", linewidth=1.0, label=f"Total Contributions (€{tot_contribution/1000:.0f}k)", alpha=0.9)
plt.axvline(np.mean(FinalValues_DH_SPY), color="blue", linestyle=":", linewidth=1.2, alpha=0.8)
plt.axvline(np.mean(FinalValues_CPPI_SPY), color="red", linestyle=":", linewidth=1.2, alpha=0.8)
plt.axvline(np.mean(FinalValuesSPY), color="dimgray", linestyle=":", linewidth=1.2, alpha=0.8)
plt.xlabel("Final Value (€)", fontsize=14)
plt.ylabel("Density (%)", fontsize=14)
plt.title("Distribution of Final Values SPY", fontsize=20, fontweight="bold")
plt.legend(loc='upper right', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x * 100:.3f}%'))
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

#%%
def describe_distribution(values, name):
    return {
        "Strategy": name,
        "Mean": np.mean(values),
        "Median": np.median(values),
        "5th %ile": np.percentile(values, 5),
        "95th %ile": np.percentile(values, 95),
        "Std": np.std(values),
        "Min": np.min(values),
        "Max": np.max(values),
        "Success Rate": np.mean(values >= tot_contribution)
    }

simulation_stats = pd.DataFrame([
    describe_distribution(FinalValuesPORT, "Optimized Portfolio"),
    describe_distribution(FinalValues_DH, "Dynamic Hybrid"),
    describe_distribution(FinalValues_CPPI, "CPPI"),
    describe_distribution(FinalValuesSPY, "SPY"),
    describe_distribution(FinalValues_DH_SPY, "DH SPY"),
    describe_distribution(FinalValues_CPPI_SPY, "CPPI SPY")
])

simulation_returns_stats = pd.DataFrame([
    describe_distribution(RetPORT_Sim, "Portfolio"),
    describe_distribution(RetSPY, "Dynamic SPY")
])

with pd.ExcelWriter("strategy_metrics_transposed.xlsx", engine="xlsxwriter") as writer:
    for df, name in [(simulation_stats, "Monte Carlo Simulation"), (simulation_returns_stats, "Returns Simulation")]:
        df_transposed = df.set_index("Strategy").T  # transpose with Strategy as column names
        df_transposed.to_excel(writer, sheet_name=name, startrow=1, header=False)

        workbook  = writer.book
        worksheet = writer.sheets[name]

        # Header formatting (for strategy names now in columns)
        header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top',
            'align': 'center', 'border': 1, 'bg_color': '#D7E4BC'
        })

        for col_num, value in enumerate(df_transposed.columns):
            worksheet.write(0, col_num + 1, value, header_format)

        # Row label formatting (now for metric names)
        index_format = workbook.add_format({
            'align': 'left', 'bold': True, 'border': 1
        })

        for row_num, idx in enumerate(df_transposed.index):
            worksheet.write(row_num + 1, 0, idx, index_format)

        # Optional: auto-fit column widths
        for i, col in enumerate(df_transposed.columns):
            max_len = max(df_transposed[col].astype(str).map(len).max(), len(str(col))) + 2
            worksheet.set_column(i + 1, i + 1, max_len)
