import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# plt.style.use('seaborn-darkgrid')

# === LOAD DATA ===
file_path=r"C:\Users\rashm\OneDrive\Desktop\PRIMA2\RUT Puts 2008.xlsx"
df = pd.read_excel(file_path)
# Now convert using explicit format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
df['Expiration'] = pd.to_datetime(df['Expiration'], format='%m/%d/%Y', errors='coerce')


# Filter data for 'IWM' and 'put' options only
# df = df[(df['Stock'] == 'IWM') & (df['Type'] == 'put')].copy()

# Helper: Get 3rd Friday of a month for expiration
def third_friday(year, month):
    # Find all Fridays in month
    from calendar import monthcalendar, SATURDAY
    cal = monthcalendar(year, month)
    fridays = [week[SATURDAY] for week in cal if week[SATURDAY] != 0]
    return datetime(year, month, fridays[2])

# Get list of unique expiration dates that are 3rd Fridays (sanity check)
third_fridays = []
years = df['Expiration'].dt.year.unique()
months = df['Expiration'].dt.month.unique()

for y in years:
    for m in months:
        try:
            d = third_friday(y, m)
            if d in df['Expiration'].values:
                third_fridays.append(d)
        except:
            pass
third_fridays = sorted(list(set(third_fridays)))
print("Unique 3rd Friday expiration dates found in data:", third_fridays)
# === 1. Identify trades: spread=20%, short strike abs(Delta) ~ 0.3, 3rd Friday expiration in following month ===

SPREAD = 0.15

def find_nearest_strike_row(df, date, target_strike):
    df_filtered = df[df['Date'] == date].copy()
    if df_filtered.empty:
        return None
    df_filtered['StrikeDiff'] = (df_filtered['Strike'] - target_strike).abs()
    nearest_row = df_filtered.sort_values(by='StrikeDiff').iloc[0]
    return nearest_row

def find_short_strike_trades(df, trade_dates, spread=SPREAD):
    trades = []
    count = 0
    
    for trade_date in trade_dates:
        df_trade = df[df['Date'] == trade_date]
        if df_trade.empty:
            continue

        # Filter candidates within delta range
        candidates = df_trade[(df_trade['Delta'].abs() >= 0.28) & (df_trade['Delta'].abs() <= 0.32)]
        if candidates.empty:
            continue

        for _, short_strike_row in candidates.iterrows():
            short_strike = short_strike_row['Strike']
            option_price_short = short_strike_row['OptionPrice']
            expiration = short_strike_row['Expiration']
            stock_price = short_strike_row['StockPrice']

            long_strike_target = short_strike * 0.85
            long_strike_row = find_nearest_strike_row(df, trade_date, long_strike_target)
            if long_strike_row is None:
                continue

            option_price_long = long_strike_row['OptionPrice']
            long_strike = long_strike_row['Strike']

            trade = {
                'OpenDate': trade_date,
                'Expiration': expiration,
                'ShortStrike': short_strike,
                'LongStrike': long_strike,
                'ShortPrice': option_price_short,
                'LongPrice': option_price_long,
                'Spread': spread,
                'StockPriceAtOpen': stock_price,
            }
            trades.append(trade)
            count += 1

    print("Number of total valid trades found:", count)
    if not trades:
        print("No trades found.")
    return trades

expiration_dates = sorted(df['Expiration'].unique())
trade_dates = []
for exp in expiration_dates:
    next_day = exp + timedelta(days=1)
    while next_day not in df['Date'].values:
        next_day += timedelta(days=1)
        if next_day > df['Date'].max():
            break
    if next_day <= df['Date'].max():
        trade_dates.append(next_day)

trades = find_short_strike_trades(df, trade_dates)

# === 2. Define Profit/Loss at Expiration for Bull Put Spread ===

def bull_put_spread_pnl(stock_price, short_strike, long_strike, net_premium):
    if stock_price >= short_strike:
        # Max profit: net premium when both puts expire worthless
        return net_premium
    elif stock_price <= long_strike:
        # Max loss: difference in strikes - net premium received
        return net_premium - (short_strike - long_strike)
    else:
        # Partial loss: the short put is in the money, long put is out
        return net_premium - (short_strike - stock_price)

for trade in trades:
    print("=================================================================")
    print("Example trade:", trade)
    expiration_date = trade['Expiration']
    stock_data = df[df['Expiration'] == expiration_date]

    if stock_data.empty:
        trade['StockPriceAtExpiration'] = None
        trade['PnL'] = None
        print("------------")
        continue

    stock_price = stock_data['StockPrice'].iloc[0]
    trade['StockPriceAtExpiration'] = stock_price

    short_strike = trade['ShortStrike']
    long_strike = trade['LongStrike']
    net_premium = trade['ShortPrice'] - trade['LongPrice']
    trade['NetPremium'] = net_premium

    trade['PnL'] = bull_put_spread_pnl(
        stock_price, short_strike, long_strike, net_premium
    )
    print("#######")
    print(trade['PnL'])

import matplotlib.pyplot as plt



valid_trades = [t for t in trades if t.get('PnL') is not None]

plt.figure(figsize=(10, 5))
plt.plot(
    [t['StockPriceAtExpiration'] for t in valid_trades],
    [t['PnL'] for t in valid_trades],
    marker='o'
)
plt.xlabel("Stock Price at Expiration")
plt.ylabel("P&L")
plt.title("Bull Put Spread: P&L at Expiration")
plt.grid(True)
plt.show()

def simulate_rolling_trades(df, initial_trades, thresholds=[0.05, 0.1]):
    rolled_trades_results = {th: [] for th in thresholds}

    for threshold in thresholds:
        count=0
        count1=0
        for trade in initial_trades:
            
            trade_chain = []
            print(f"#########Processing trade with threshold {threshold}: {trade['OpenDate']} ")
            open_date = trade['OpenDate']
            expiration = trade['Expiration']
            short_strike = trade['ShortStrike']
            long_strike = trade['LongStrike']
            spread = trade['Spread']
            net_premium = trade['ShortPrice'] - trade['LongPrice']

            # Step 1: Check if stock price falls below ShortStrike before expiration
            df_initial = df[(df['Date'] >= open_date) & (df['Date'] < expiration)]
            condition1 = df_initial['StockPrice'] <= short_strike

            if condition1.any():
                # Get first roll date
                print(f"condition1======{open_date}")
                roll_date = df_initial[condition1]['Date'].iloc[0]
                stock_price_on_roll = df_initial[condition1]['StockPrice'].iloc[0]
                pnl = bull_put_spread_pnl(np.array([stock_price_on_roll]), short_strike, long_strike, net_premium)
                trade_chain.append({
                    'OpenDate': open_date,
                    'Expiration': expiration,
                    'ShortStrike': short_strike,
                    'LongStrike': long_strike,
                    'NetPremium': net_premium,
                    'CloseDate': roll_date,
                    'FinalStockPrice': stock_price_on_roll,
                    'PnL': float(pnl)
                })

                # New trade with same Short Strike, Long Strike = 0.9 * Short Strike, new expiration
                new_long_strike = round(0.9 * short_strike, 2)
                new_short_strike = short_strike  # remains same

                # roll_expirations = sorted([d for d in third_fridays if d >= roll_date + pd.DateOffset(months=1)])
                roll_expirations = sorted([d for d in df['Expiration'] if d >= roll_date + pd.DateOffset(months=3)])
                if not roll_expirations:
                    continue
                print(f"roll_expirations found--------{roll_date},,{roll_expirations[0]}")
                new_expiration = roll_expirations[0]

                # Always fetch fresh data from original df (not from reused df_new_trade)
                df_new_trade = df[(df['Date'] >= roll_date) & (df['Expiration'] >= new_expiration)].copy()

            

                short_row = df[df['Strike'] <= new_short_strike ]
                long_row = df[df['Strike'] >= new_long_strike]

                if short_row.empty or long_row.empty:
                    count=count+1
                    print(f"Short or Long strike not found for new trade on {roll_date}. Skipping trade.")
                    continue

                new_short_price = short_row['OptionPrice'].iloc[0]
                new_long_price = long_row['OptionPrice'].iloc[0]
                new_net_premium = new_short_price - new_long_price
                
                # Step 2: Track new trade for threshold rolling
                current_open = roll_date
                current_exp = new_expiration
                current_short = new_short_strike
                current_long = new_long_strike
                current_net = new_net_premium

                while True:
                    df_track = df_new_trade[(df_new_trade['Date'] > current_open) & (df_new_trade['Date'] <= current_exp)]
                    trigger_price = (1 - threshold) * current_long
                    condition2 = df_track['StockPrice'] <= trigger_price

                    if not condition2.any():
                     
                        if df_track.empty:
                            print(f"No data to track between {current_open} and {current_exp}. Skipping final PnL calc.")
                            break  # or continue, depending on context
                        print(f"condition2 not found for threshold {threshold}, rolling trades complete.")
                        final_price = df_track['StockPrice'].iloc[-1]
                        pnl = bull_put_spread_pnl(np.array([final_price]), current_short, current_long, current_net)
                       
                        break
                    count=1
                    count1=count1+1
                    print(f"condition2 found for threshold {threshold}, rolling trade on {current_open} with stock price {df_track['StockPrice'].iloc[0]}")
                    roll_date2 = df_track[condition2]['Date'].iloc[0]
                    stock_price2 = df_track[condition2]['StockPrice'].iloc[0]
                    pnl = bull_put_spread_pnl(np.array([stock_price2]), current_short, current_long, current_net)

                    trade_chain.append({
                        'OpenDate': current_open,
                        'Expiration': current_exp,
                        'ShortStrike': current_short,
                        'LongStrike': current_long,
                        'NetPremium': current_net,
                        'CloseDate': roll_date2,
                        'FinalStockPrice': stock_price2,
                        'PnL': float(pnl)
                    })

                    # if method == 'method1':
                    new_long = round(max((1 + 0.5 * threshold) * stock_price2, 1.025 * stock_price2), 2)
                    
                    new_long = max(new_long,round(1.025 * stock_price2, 2))
                    new_short = round(new_long / (1 - spread), 2)

                    df_new_trade2 = df[(df['Date'] >= roll_date2) & (df['Expiration'] == current_exp)].copy()
                    

                    short_row2 = df[df['Strike'] <= new_short]
                    long_row2 = df[df['Strike'] >= new_long]

                    if short_row2.empty or long_row2.empty:
                        print(f"Short or Long strike not found for new trade on {roll_date2}. Skipping trade--2.")
                        break
               
                    new_short_price2 = short_row2['OptionPrice'].iloc[0]
                    new_long_price2 = long_row2['OptionPrice'].iloc[0]
                    new_net = new_short_price2 - new_long_price2

                    current_open = roll_date2
                    current_short = new_short
                    current_long = new_long
                    current_net = new_net

                rolled_trades_results[threshold].append(trade_chain)
          
            print(f"condition1 not found for threshold {threshold}, rolling trades complete.")
        print(f"----{count1}")
    return rolled_trades_results

def extract_pnl_and_expired(trades_list, threshold):
    pnl_all = []
    expired_below_long = []
    for trade_chain in trades_list:
        for t in trade_chain:
            pnl_all.append({
                'Threshold': threshold,
                'PnL':float(t['PnL'])
            })
            expired_below_long.append({
                'Threshold': threshold,
                'ExpiredBelowLong': int(t['FinalStockPrice'] < t['LongStrike'])
            })

    return pnl_all, expired_below_long

def fix_pnl(val):
    if isinstance(val, np.ndarray):
        # Example: take the first element as scalar
        return val[0]
    return val
def main():
    # Load and filter data
    import matplotlib.pyplot as plt
    print(plt.style.available)

    # Prepare expiration dates and trade dates
    expiration_dates = sorted(df['Expiration'].unique())
    trade_dates = []
    for exp in expiration_dates:
        next_day = exp + timedelta(days=1)
        while next_day not in df['Date'].values:
            next_day += timedelta(days=1)
            if next_day > df['Date'].max():
                break
        if next_day <= df['Date'].max():
            trade_dates.append(next_day)

    trades = find_short_strike_trades(df, trade_dates)

    if not trades:
        print("No trades found.")
        return

    rolled_results = simulate_rolling_trades(df, trades, thresholds=[0.05, 0.1])
    
    for i in rolled_results:
        print(f"Threshold {i} has {len(rolled_results[i])} trade chains.")
        print(f"=====----------------{rolled_results[i]}")
        
    # Extract and plot PnL
    all_pnl = []
    all_expired = []

    for threshold, trade_chains in rolled_results.items():
       pnl, expired = extract_pnl_and_expired(trade_chains, threshold)
       print(f"Threshold: {threshold}, PnL count: {len(pnl)}, Expired count: {len(expired)}")
    #    print("PnL data:", pnl)
    #    print("Expired data:", expired)
       all_pnl.extend(pnl)
       all_expired.extend(expired)


    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

# Flatten rolled trades into DataFrame
    all_trades = []
 
    for threshold, chains in rolled_results.items():
     for chain in chains:
        for trade in chain:
            all_trades.append({
                'Threshold': threshold,
                'PnL (%)': 100 * trade['PnL'] / (trade['ShortStrike'] - trade['LongStrike'])  # normalized PnL
            })

    df_all_trades = pd.DataFrame(all_trades)

# Box plot of PnL
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_all_trades, x='Threshold', y='PnL (%)')
    plt.title("PnL (%) of All Trades (Intermediate + Final) per Threshold")
    plt.grid(True)
    plt.tight_layout()
    plt.show()        
    final_trades_summary = []
    for threshold, chains in rolled_results.items():
     expired_below_long = 0
     total = 0
     for chain in chains:
        for final_trade in chain:
          if final_trade['FinalStockPrice'] <= final_trade['LongStrike']:
             expired_below_long += 1
          total += 1
     percent_below = 100 * expired_below_long / total
     final_trades_summary.append({'Threshold': threshold, 'Breach (%)': percent_below})

    df_summary = pd.DataFrame(final_trades_summary)

# Bar plot
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_summary, x='Threshold', y='Breach (%)', hue='Threshold', palette='Blues_d', legend=False)
    plt.title("Percent of Final Trades Expired with StockPrice ≤ Long Strike")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    pnl_df = pd.DataFrame(all_pnl)
    expired_df = pd.DataFrame(all_expired)

    pnl_df['PnL'] = pnl_df['PnL'].apply(fix_pnl)
    import matplotlib.pyplot as plt
    import seaborn as sns

# (a) Collect PnLs (% returns)
    pnl_data = []

# (b) % of trades with StockPrice <= LongStrike at Expiration
    expiration_breaches = {threshold: {'count': 0, 'total': 0} for threshold in rolled_results}

    for threshold, trade_chains in rolled_results.items():
     for chain in trade_chains:
        if not chain:
            continue

        for t in chain:
            open_premium = abs(t['NetPremium'])
            pnl_percent = 100 * t['PnL']/ open_premium if open_premium != 0 else 0
            pnl_data.append({'Threshold': threshold, 'PnL (%)': pnl_percent})

        # Get last trade in chain (final rolled trade)
        final_trade = chain[-1]
        final_stock = final_trade['FinalStockPrice']
        final_long_strike = final_trade['LongStrike']

        expiration_breaches[threshold]['total'] += 1
        if final_stock <= final_long_strike:
            expiration_breaches[threshold]['count'] += 1

    # Create DataFrame for seaborn
    pnl_df = pd.DataFrame(pnl_data)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=pnl_df, x='Threshold', y='PnL (%)')
    plt.title('Profit (%) Distribution vs Threshold')
    plt.grid(True)
    plt.show()
    
    
    
    
    for trade in trades:
     print("=================================================================")
     print("Example trade:", trade)
     expiration_date = trade['Expiration']
     stock_data = df[df['Expiration'] == expiration_date]

     if stock_data.empty:
        trade['StockPriceAtExpiration'] = None
        trade['PnL'] = None
        print("------------")
        continue

     stock_price = stock_data['StockPrice'].iloc[0]
     trade['StockPriceAtExpiration'] = stock_price
 
     short_strike = trade['ShortStrike']
     long_strike = trade['LongStrike']
     net_premium = trade['ShortPrice'] - trade['LongPrice']
     trade['NetPremium'] = net_premium
 
     trade['PnL'] = bull_put_spread_pnl(
         stock_price, short_strike, long_strike, net_premium
     )
    import pandas as pd
    import numpy as np
    
    # === Step 1: Convert trade list to DataFrame ===
    df_trades = pd.DataFrame(trades)
    df_trades = df_trades.dropna(subset=['PnL'])  # Remove incomplete trades
    
    # === Step 2: Build Portfolio Value over time ===
    initial_value = 100000
    portfolio_values = [initial_value]
    dates = [df_trades.iloc[0]['OpenDate']]
    
    for _, trade in df_trades.iterrows():
        pnl = trade['PnL'] * 100  # Assuming 100 lot size
        new_value = portfolio_values[-1] + pnl
        portfolio_values.append(new_value)
        dates.append(trade['Expiration'])
    
    df_portfolio = pd.DataFrame({'Date': dates, 'PortfolioValue': portfolio_values})
    df_portfolio.set_index('Date', inplace=True)
    df_portfolio = df_portfolio.groupby(df_portfolio.index).mean()
    df_portfolio = df_portfolio.asfreq('D').ffill()

    
    # === Step 3: Calculate Daily Returns ===
    df_portfolio['DailyReturn'] = df_portfolio['PortfolioValue'].pct_change()
    
    # === Step 4: Maximum Drawdown ===
    roll_max = df_portfolio['PortfolioValue'].cummax()
    drawdown = (df_portfolio['PortfolioValue'] - roll_max) / roll_max
    max_drawdown = drawdown.min()
    
    # === Step 5: Average Maximum Drawdown (1-Year Rolling) ===
    def max_dd_1yr(window):
        roll_max = window.cummax()
        dd = (window - roll_max) / roll_max
        return dd.min()
    
    rolling_dd = df_portfolio['PortfolioValue'].rolling(window=252).apply(max_dd_1yr)
    avg_rolling_max_dd = rolling_dd.mean()
    
    # === Step 6: Volatility (Std Dev of Daily Returns) ===
    daily_vol = df_portfolio['DailyReturn'].std()
    annual_vol = daily_vol * np.sqrt(252)
    
    # === Step 7: Value at Risk (Historical VaR 95%) ===
    var_95 = df_portfolio['DailyReturn'].quantile(0.05)
    
    # === Step 8: Sharpe Ratio ===
    risk_free_rate = 0.02
    avg_daily_return = df_portfolio['DailyReturn'].mean()
    annual_return = avg_daily_return * 252
    sharpe_ratio = (annual_return - risk_free_rate) / annual_vol
    
    # === Step 9: Sortino Ratio ===
    downside_returns = df_portfolio['DailyReturn'][df_portfolio['DailyReturn'] < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - risk_free_rate) / downside_std
    
    # === Step 10: Calmar Ratio ===
    start_val = df_portfolio['PortfolioValue'].iloc[0]
    end_val = df_portfolio['PortfolioValue'].iloc[-1]
    n_years = (df_portfolio.index[-1] - df_portfolio.index[0]).days / 365
    cagr = (end_val / start_val) ** (1 / n_years) - 1
    calmar_ratio = cagr / abs(max_drawdown)
    
    # === Print All Metrics ===
    print("======== RISK METRICS ========")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Avg Max Drawdown (1-Year Rolling): {avg_rolling_max_dd:.2%}")
    print(f"Annualized Volatility: {annual_vol:.2%}")
    print(f"Historical VaR (95%): {var_95:.2%}")
    
    print("\n======== REWARD METRICS ========")
    print(f"Annual Return (CAGR): {cagr:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")
    
       
    
    
if __name__ == '__main__':
    main()