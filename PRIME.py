import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# plt.style.use('seaborn-darkgrid')

# === LOAD DATA ===
df = pd.read_csv(r"C:\Users\rashm\OneDrive\Desktop\PRIMA2\Prima\Prima\IWM Puts 2024.csv", parse_dates=['Date', 'Expiration'])

# Filter data for 'IWM' and 'put' options only
# df = df[(df['Stock'] == 'IWM') & (df['Type'] == 'put')].copy()

# Helper: Get 3rd Friday of a month for expiration
def third_friday(year, month):
    # Find all Fridays in month
    from calendar import monthcalendar, FRIDAY
    cal = monthcalendar(year, month)
    fridays = [week[FRIDAY] for week in cal if week[FRIDAY] != 0]
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

# === 1. Identify trades: spread=20%, short strike abs(Delta) ~ 0.3, 3rd Friday expiration in following month ===

SPREAD = 0.2

def find_nearest_strike_row(df, date, target_strike):
    df_filtered = df[df['Date'] == date].copy()
    if df_filtered.empty:
        return None
    df_filtered['StrikeDiff'] = (df_filtered['Strike'] - target_strike).abs()
    nearest_row = df_filtered.sort_values(by='StrikeDiff').iloc[0]
    return nearest_row

def find_short_strike_trades(df, trade_dates, spread=SPREAD):
    trades = []
    count=0
    for trade_date in trade_dates:
        next_month = trade_date + pd.DateOffset(months=1)
        expiration = third_friday(next_month.year, next_month.month)

        df_trade = df[(df['Date'] == trade_date) & (df['Expiration'] == expiration)]
        if df_trade.empty:
            continue
        
        stock_price = df_trade['StockPrice'].iloc[0]
        candidates = df_trade[(df_trade['Delta'].abs() >= 0.28) & (df_trade['Delta'].abs() <= 0.32)]
        if candidates.empty:
            continue
        count+=1
        candidates = candidates.copy()
        print("Candidates for trade date", trade_date, ":", candidates)
        candidates['StrikeDiff'] = (candidates['Strike'] - stock_price).abs()
        short_strike_row = candidates.sort_values('StrikeDiff').iloc[0]
        short_strike = short_strike_row['Strike']
        option_price_short = short_strike_row['OptionPrice']

        long_strike = short_strike *0.8
        long_strike_row = find_nearest_strike_row(df, trade_date, long_strike)
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
    print("Number of trades found:", count)
    
    print("Number of trades found:", len(trades))
    if not trades:
        print("No trades found.")
    return trades




#here we are finding open trade dates which are afger expiartion and we pass those dates defined as trade_dates to function find_short_strike_trades
# that functon will return values for each open trade date with the expiration date,long and short strike prices, option prices and the spread
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



# # === 3. Plot example P/L curve for one trade ===
# example_trade = {
#         'ShortStrike': 200,
#         'LongStrike': 180,
#         'ShortPrice': 5,
#         'LongPrice': 2
#     }
# if trades:
#     example_trade = trades[0]
# else:
#     print("No trades found.")
   



for trade in trades:
    print("Example trade:", trade)
    expiration_date = trade['Expiration']
    stock_data = df[df['Date'] == expiration_date]

    if stock_data.empty:
        trade['StockPriceAtExpiration'] = None
        trade['PnL'] = None
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

import matplotlib.pyplot as plt

# Filter out trades with missing PnL
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


# === 4. Rolling trades based on rules given (simplified prototype) ===

# We'll simulate rolling trades on threshold triggers and update strike prices accordingly

def simulate_rolling_trades(df, initial_trades, thresholds=[0.05, 0.1]):
    rolled_trades_results = {th: [] for th in thresholds}
    for threshold in thresholds:
        for trade in initial_trades:
            # Setup initial trade parameters
            current_trade = trade.copy()
            open_date = current_trade['OpenDate']
            expiration = current_trade['Expiration']
            short_strike = current_trade['ShortStrike']
            long_strike = current_trade['LongStrike']
            spread = current_trade['Spread']

            net_premium = current_trade['ShortPrice'] - current_trade['LongPrice']

            # Track trade over time from open_date to expiration

            # For correction mode: two methods for Long Strike calculation
            # Method 1: Long Strike = (1 + 0.5*Threshold) * StockPrice
            # Method 2: Long Strike = 1.025 * StockPrice

            # Track rolling trades
            rolled = []
            current_open_date = open_date
            current_expiration = expiration
            current_short_strike = short_strike
            current_long_strike = long_strike
            current_net_premium = net_premium
            current_spread = spread

            method_results = {'method1': [], 'method2': []}

            for method in ['method1', 'method2']:
                trades_chain = []
                open_d = current_open_date
                exp_d = current_expiration
                short_s = current_short_strike
                long_s = current_long_strike
                net_p = current_net_premium
                spread_ = current_spread

                df_period = df[(df['Date'] >= open_d) & (df['Date'] <= exp_d)].copy()
                rolled_trade_active = True
                print(f"def period {df_period}")
                print(f"stock price open {df_period['StockPrice']}")
                print(f"LONG STRINKE {long_s}")
                print(f"short strike {short_s}")
            
                while rolled_trade_active and not df_period.empty:
                    # Find if stock price drops below (1-threshold)*long strike
                    trigger_price = (1 - threshold) * long_s
                    condition = df_period['StockPrice'] <= long_s
                    print(f"Trigger price: {trigger_price}, Condition: {condition.any()}")

                    if not condition.any():
                        # No trigger, trade lasts until expiration
                        final_stock_price = df_period['StockPrice'].iloc[-1]
                        # Calculate final P/L at expiration
                        pnl = bull_put_spread_pnl(np.array([final_stock_price]), short_s, long_s, net_p)
                        trades_chain.append({
                            'OpenDate': open_d,
                            'Expiration': exp_d,
                            'ShortStrike': short_s,
                            'LongStrike': long_s,
                            'NetPremium': net_p,
                            'FinalStockPrice': final_stock_price,
                            'PnL': pnl
                        })
                        rolled_trade_active = False
                        print(f"Trade closed at expiration on {exp_d} with stock price {final_stock_price}")
                    else:
                        # Roll trade on first occurrence
                        roll_date = df_period[condition]['Date'].iloc[0]
                        stock_price_on_roll = df_period[condition]['StockPrice'].iloc[0]
                        print(f"Rolling trade on {roll_date} with stock price {stock_price_on_roll}")

                        # Close current trade on roll_date
                        # Calculate P/L at roll date
                        pnl = bull_put_spread_pnl(np.array([stock_price_on_roll]), short_s, long_s, net_p)

                        trades_chain.append({
                            'OpenDate': open_d,
                            'Expiration': exp_d,
                            'ShortStrike': short_s,
                            'LongStrike': long_s,
                            'NetPremium': net_p,
                            'FinalStockPrice': stock_price_on_roll,
                            'PnL': pnl,
                            'CloseDate': roll_date
                        })

                        # Open new trade same spread and expiration >= 3 months away
                        roll_expiration_candidates = sorted([d for d in third_fridays if d >= roll_date + pd.DateOffset(months=3)])
                        if len(roll_expiration_candidates) == 0:
                            # No further expiration dates; stop rolling
                            rolled_trade_active = False
                            break
                        new_expiration = roll_expiration_candidates[0]

                        # Calculate new long strike per method
                        if method == 'method1':
                            new_long_strike = max((1 + 0.5*threshold)*stock_price_on_roll, 1.025*stock_price_on_roll)
                        else:
                            new_long_strike = 1.025*stock_price_on_roll

                        # Round strikes sensibly
                        new_long_strike = round(new_long_strike, 2)
                                                # Estimate new short strike given spread
                        new_short_strike = round(new_long_strike / (1 - spread_), 2)

                        # Get new option prices for short and long
                        # df_new_trade = df[(df['Date'] == roll_date) & (df['Expiration'] == new_expiration)]
                        df_new_trade = df[(df['Date'] <= roll_date) & (df['Expiration'] == new_expiration)]
                        if not df_new_trade.empty:
                           df_new_trade = df_new_trade.sort_values(by='Date', ascending=False).iloc[0:1]

                        short_option_row = df_new_trade[df_new_trade['Strike'] == new_short_strike]
                        long_option_row = df_new_trade[df_new_trade['Strike'] == new_long_strike]

                        if short_option_row.empty or long_option_row.empty:
                            # Skip if prices not available
                            rolled_trade_active = False
                            break

                        new_short_price = short_option_row['OptionPrice'].iloc[0]
                        new_long_price = long_option_row['OptionPrice'].iloc[0]
                        new_net_premium = new_short_price - new_long_price

                        # Update for next iteration
                        open_d = roll_date
                        exp_d = new_expiration
                        short_s = new_short_strike
                        long_s = new_long_strike
                        net_p = new_net_premium

                        df_period = df[(df['Date'] >= open_d) & (df['Date'] <= exp_d)].copy()

                method_results[method] = trades_chain
            rolled_trades_results[threshold].append(method_results)
    return rolled_trades_results

# Run simulation on first 5 trades for demo (to limit compute)
rolled_results = simulate_rolling_trades(df, trades, thresholds=[0.05, 0.1])

# === 5. Analyze and plot results ===

def extract_pnl_and_expired(trades_list):
    pnl_all = []
    expired_below_long = []
    for trade_chain in trades_list:
        for method in ['method1', 'method2']:
            trades = trade_chain[method]
            for t in trades:
                pnl_all.append({'Threshold': threshold,
                                'Method': method,
                                'PnL': t['PnL']})
                expired_below_long.append({'Threshold': threshold,
                                          'Method': method,
                                          'ExpiredBelowLong': int(t['FinalStockPrice']< t['LongStrike'])})
    return pnl_all, expired_below_long

all_pnl = []
all_expired = []
for threshold, trade_chains in rolled_results.items():
    pnl, expired = extract_pnl_and_expired(trade_chains)
    print(f"Threshold: {threshold}, PnL count: {len(pnl)}, Expired count: {len(expired)}")
    print("PnL data:", pnl) 
    print("Expired data:", expired)
    print("Threshold:", threshold)
    all_pnl.extend(pnl)
    all_expired.extend(expired)


pnl_df = pd.DataFrame(all_pnl)
expired_df = pd.DataFrame(all_expired)
#save to csv
pnl_df.to_csv("pnl_df.csv", index=False)
expired_df.to_csv("expired_df.csv", index=False)

# Boxplot of profits by threshold and method
plt.figure(figsize=(12,6))
print("Columns in pnl_df:", pnl_df.columns)
print(pnl_df.head())

import numpy as np

print("Checking 'Threshold' for ndarray values:", pnl_df['Threshold'].apply(lambda x: isinstance(x, np.ndarray)).sum())
print("Checking 'PnL' for ndarray values:", pnl_df['PnL'].apply(lambda x: isinstance(x, np.ndarray)).sum())
print("Checking 'Method' for ndarray values:", pnl_df['Method'].apply(lambda x: isinstance(x, np.ndarray)).sum())

mask = pnl_df[['Threshold', 'PnL', 'Method']].applymap(lambda x: isinstance(x, np.ndarray))
# print("Rows with ndarray values in any column among Threshold, PnL, Method:")
# print(pnl_df[mask.any(axis=1)])
import numpy as np

def fix_pnl(val):
    if isinstance(val, np.ndarray):
        # Example: take the first element as scalar
        return val[0]
    return val

pnl_df['PnL'] = pnl_df['PnL'].apply(fix_pnl)
sns.boxplot(x='Threshold', y='PnL', hue='Method', data=pnl_df)
plt.title('Profit (%) Distribution for Rolled Trades by Threshold and Method')
plt.ylabel('Profit / Loss')
plt.show()

# Percentage of trades expired below Long Strike by threshold and method
expired_summary = expired_df.groupby(['Threshold', 'Method'])['ExpiredBelowLong'].mean().reset_index()
expired_summary['PercentExpiredBelowLong'] = expired_summary['ExpiredBelowLong'] * 100
import numpy as np

def fix_pnl(val):
    if isinstance(val, np.ndarray):
        # Example: take the first element as scalar
        return val[0]
    return val

pnl_df['PnL'] = pnl_df['PnL'].apply(fix_pnl)

plt.figure(figsize=(12,6))
sns.barplot(x='Threshold', y='PercentExpiredBelowLong', hue='Method', data=expired_summary)
plt.title('Percentage of Trades Expired with Stock Price ≤ Long Strike')
plt.ylabel('Percentage (%)')
plt.show()

# === 6. Feature importance placeholder for training set inclusion ===

def feature_selection_placeholder(df):
    """
    Dummy method to assess parameters for inclusion in training set.
    You can use correlation with PnL, or information gain with classification target.
    """
    features = ['IV', 'Delta', 'Gamma', 'Theta', 'Vega', 'Volume', 'OpenInterest']
    # Compute correlations with OptionPrice as proxy
    correlations = df[features + ['OptionPrice']].corr()['OptionPrice'].sort_values(ascending=False)
    print("Feature correlation with OptionPrice:")
    print(correlations)
    # Return features with abs(correlation) > 0.2 as example
    selected_features = correlations[correlations.abs() > 0.2].index.tolist()
    selected_features.remove('OptionPrice')
    return selected_features

selected_feats = feature_selection_placeholder(df)
print("Selected features for training set:", selected_feats)

# === 7. Technical indicators & Threshold intuition ===

# Simple Moving Average (SMA) and RSI calculations for stock price

def compute_technical_indicators(df):
    df = df.sort_values('Date')
    df['SMA_10'] = df['StockPrice'].rolling(window=10).mean()
    delta = df['StockPrice'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + RS))
    return df

df_tech = compute_technical_indicators(df)

# Visual inspection of stock price vs threshold triggers can be done manually with plots:
plt.figure(figsize=(12,6))
plt.plot(df_tech['Date'], df_tech['StockPrice'], label='Stock Price')
plt.plot(df_tech['Date'], df_tech['SMA_10'], label='SMA 10')
plt.title('Stock Price & SMA')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df_tech['Date'], df_tech['RSI_14'], label='RSI 14')
plt.axhline(30, linestyle='--', color='red')
plt.axhline(70, linestyle='--', color='green')
plt.title('RSI Indicator')
plt.legend()
plt.show()

print("Use these plots to inform rolling threshold decisions based on momentum and trend.")
def fix_pnl(val):
    if isinstance(val, np.ndarray):
        # Example: take the first element as scalar
        return val[0]
    return val
def main():
    # Load and filter data
    import matplotlib.pyplot as plt
    print(plt.style.available)

    # df = pd.read_csv(r"C:\Users\rashm\OneDrive\Desktop\PRIMA2\IWM Puts 2023.csv", parse_dates=['Date', 'Expiration'])
    # df = df[(df['Stock'] == 'IWM') & (df['Type'] == 'put')].copy()

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

    rolled_results = simulate_rolling_trades(df, trades[:5], thresholds=[0.05, 0.1])
   
    # Extract and plot PnL
    all_pnl = []
    all_expired = []
    for threshold, trade_chains in rolled_results.items():
        pnl, expired = extract_pnl_and_expired(trade_chains)
        all_pnl.extend(pnl)
        all_expired.extend(expired)

    pnl_df = pd.DataFrame(all_pnl)
    expired_df = pd.DataFrame(all_expired)

    import numpy as np

    # print("Checking 'Threshold' for ndarray values:", pnl_df['Threshold'].apply(lambda x: isinstance(x, np.ndarray)).sum())
    # print("Checking 'PnL' for ndarray values:", pnl_df['PnL'].apply(lambda x: isinstance(x, np.ndarray)).sum())
    # print("Checking 'Method' for ndarray values:", pnl_df['Method'].apply(lambda x: isinstance(x, np.ndarray)).sum())

    # mask = pnl_df[['Threshold', 'PnL', 'Method']].applymap(lambda x: isinstance(x, np.ndarray))
    # print("Rows with ndarray values in any column among Threshold, PnL, Method:")
    # print(pnl_df[mask.any(axis=1)])
    import numpy as np



    pnl_df['PnL'] = pnl_df['PnL'].apply(fix_pnl)

    # plt.figure(figsize=(12,6))
    # sns.boxplot(x='Threshold', y='PnL', hue='Method', data=pnl_df)
    # plt.title('Profit (%) Distribution for Rolled Trades by Threshold and Method')
    # plt.ylabel('Profit / Loss')
    # plt.show()

    # expired_summary = expired_df.groupby(['Threshold', 'Method'])['ExpiredBelowLong'].mean().reset_index()
    # expired_summary['PercentExpiredBelowLong'] = expired_summary['ExpiredBelowLong'] * 100

    
    # print(expired_summary['PercentExpiredBelowLong'].describe())
    # print(expired_summary)

    # plt.figure(figsize=(12,6))
    # pnl_df['PnL'] = pnl_df['PnL'].apply(fix_pnl)
    # sns.barplot(x='Threshold', y='PercentExpiredBelowLong', hue='Method', data=expired_summary)
    # plt.title('Percentage of Trades Expired Below Long Strike by Threshold and Method')
    # plt.ylabel('Percentage (%)')
    # plt.show()

if __name__ == '__main__':
    main()