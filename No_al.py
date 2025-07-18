import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
portfolio_value = 100000  # Initial portfolio value
# plt.style.use('seaborn-darkgrid')

# === LOAD DATA ===
# file_path=r"C:\Users\rashm\OneDrive\Desktop\PRIMA2\RUT Puts 2024.xlsx"
# df = pd.read_excel(file_path)

from sqlalchemy import create_engine

server = 'localhost'
database = 'OptionsDataDB'
driver = 'ODBC Driver 17 for SQL Server'

connection_string = f"mssql+pyodbc://localhost/{database}?driver={driver.replace(' ', '+')}&trusted_connection=yes"
engine = create_engine(connection_string)
print(f"Reading")
import pandas as pd
df = pd.read_sql(
    "SELECT Date, StockPrice, Expiration, Strike, OptionPrice, Delta FROM OptionsData",
    engine
)
print(f"Reading========")
# Now convert using explicit format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
df['Expiration'] = pd.to_datetime(df['Expiration'], format='%m/%d/%Y', errors='coerce')


from calendar import monthcalendar, FRIDAY

def get_third_friday_next_month(open_date):
    year = open_date.year
    month = open_date.month

    # Move to next month
    if month == 12:
        month = 1
        year += 1
    else:
        month += 1

    cal = monthcalendar(year, month)
    fridays = [week[FRIDAY] for week in cal if week[FRIDAY] != 0]
    return datetime(year, month, fridays[2])


# Get list of unique expiration dates that are 3rd Fridays (sanity check)
third_fridays = []
years = df['Expiration'].dt.year.unique()
months = df['Expiration'].dt.month.unique()

SPREAD = 0.15

def find_nearest_strike_row(df, date, exp,sp, target_strike):
    df_filtered = df[(df['Date'] == date) & (df['Expiration']==exp) &(df['StockPrice']==sp)].copy()
    if df_filtered.empty:
        return None
    cond=df_filtered['Strike']>= target_strike
    df_filtered = df_filtered[cond]
    if df_filtered.empty:
        return None
    df_filtered['StrikeDiff'] = (df_filtered['Strike'] - target_strike).abs()
    nearest_row = df_filtered.sort_values(by='StrikeDiff').iloc[0]
    return nearest_row
def find_short_strike_trades(df, trade_dates, spread=SPREAD):
    trades = []
    count = 0

    for trade_date in trade_dates:
        df_trade = df[(df['Date'] == trade_date)]
        if df_trade.empty:
            continue

        candidates = df_trade[(df_trade['Delta'].abs() >= 0.19) & (df_trade['Delta'].abs() <= 0.21)]
        if candidates.empty:
            continue

        count += 1
        stock_price = df_trade['StockPrice'].iloc[0]
        candidates = candidates.copy()
        candidates['StrikeDiff'] = (candidates['Strike'] - stock_price).abs()
        short_strike_row = candidates.sort_values('StrikeDiff').iloc[0]
        short_strike = short_strike_row['Strike']
        option_price_short = short_strike_row['OptionPrice']

        long_strike = short_strike * 0.9
        long_strike_row = find_nearest_strike_row(df, trade_date, short_strike_row['Expiration'], stock_price, long_strike)
        if long_strike_row is None:
            continue

        option_price_long = long_strike_row['OptionPrice']
        long_strike = long_strike_row['Strike']
        close_date = get_third_friday_next_month(trade_date)  # <-- NEW LINE

        trade = {
            'OpenDate': trade_date,
            'CloseDate': close_date,  # <-- NEW LINE
            'Expiration': short_strike_row['Expiration'],
            'ShortStrike': short_strike_row['Strike'],
            'LongStrike': long_strike_row['Strike'],
            'ShortPrice': option_price_short,
            'LongPrice': option_price_long,
            'Spread': spread,
            'StockPriceAtOpen': stock_price,
            'ShortDelta': short_strike_row['Delta'],
            'LongDelta': long_strike_row['Delta'],
        }

        trades.append(trade)

    print("Number of trades found:", len(trades))
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

for trade in trades:
    print("=================================================================")
    print("Example trade:", trade)   

import matplotlib.pyplot as plt
def simulate_rolling_trades(df, initial_trades, thresholds=[0.1]):
    rolled_trades_results = {th: [] for th in thresholds}

    for threshold in thresholds:
        
        portfolio_value=100000  # Reset portfolio value for each threshold
        
        trade_chain = []
        
        for trade in initial_trades:
            net_credit=0
            net_debit=0
            init_portfolio_value = portfolio_value  # Store initial portfolio value for this trade
            trade_chain = []
            print(f"#########Processing trade with threshold {threshold}: {trade['OpenDate']} ")
            open_date = trade['OpenDate']
            expiration = trade['Expiration']
            short_strike = trade['ShortStrike']
            long_strike = trade['LongStrike']
            net_premium1 = trade['ShortPrice'] - trade['LongPrice']
            buffer= (trade['StockPriceAtOpen']- short_strike)/ short_strike
            Close= trade['CloseDate']
            # Step 1: Check if stock price falls below ShortStrike before expiration
            spread= (short_strike-long_strike)/short_strike
            spread= round(spread*100,2)
            contracts= float((init_portfolio_value*0.25)/(spread * short_strike))
            contracts=round(contracts,2)
            net_credit =net_premium1 * contracts *100
            p1=portfolio_value
            portfolio_value += net_credit
            
            
            print("========================================Entered")
            expiration_date = trade['Expiration']
             
            stock_data = df[(df['Date'] == Close) & (df['Strike'] == short_strike) ]
            if not stock_data.empty:
                max_date = stock_data['Date'].max()
                   
                   # Step 3: Filter rows having this max_date
                max_date_rows = stock_data[stock_data['Date'] == max_date]
                   
                   # Step 4: Among those, pick the one with the minimum Expiration
                result_row = max_date_rows.loc[max_date_rows['Expiration'].idxmin()]
                stock_data= result_row
            if stock_data.empty:
                # Step 1: Filter rows where Date < expiration_date and Strike == short_strike
               filtered_df = df[
                   (df['Date'] < Close) &
                   (df['Strike'] == short_strike)
               ]
               result_row = None
               # Step 2: If not empty, proceed
               if not filtered_df.empty:
                   # Find the latest date
                   max_date = filtered_df['Date'].max()
                   
                   # Step 3: Filter rows having this max_date
                   max_date_rows = filtered_df[filtered_df['Date'] == max_date]
                   
                   # Step 4: Among those, pick the one with the minimum Expiration
                   result_row = max_date_rows.loc[max_date_rows['Expiration'].idxmin()]
                   stock_data= result_row
               

              
            
            #    print(f"{stock_data}===================================")
            s1= df[((df['Date'])==stock_data['Date'])  &  (df['Expiration']==stock_data['Expiration']) & (df['Strike'] == long_strike) ].copy()
            s1=s1.tail(1)
            #   print(s1,"=================")
            s_s_p=stock_data['OptionPrice'] if not stock_data.empty else 0
            l_s_p= s1['OptionPrice'].iloc[0] if not s1.empty else 0
            s_s_p=0 if s1.empty else s_s_p
            net_debit= (s_s_p - l_s_p) * contracts * 100
        
             # Get the first row for the expiration date
            stock_price = stock_data['StockPrice']
            c_d=stock_data['Date']
            if Close>expiration_date:
              stock_data = df[(df['Date'] == expiration_date) & (df['Strike'] == short_strike) ]
              if not stock_data.empty:
                max_date = stock_data['Date'].max()
                   
                   # Step 3: Filter rows having this max_date
                max_date_rows = stock_data[stock_data['Date'] == max_date]
                   
                   # Step 4: Among those, pick the one with the minimum Expiration
                result_row = max_date_rows.loc[max_date_rows['Expiration'].idxmin()]
                stock_data= result_row
              if stock_data.empty:
                # Step 1: Filter rows where Date < expiration_date and Strike == short_strike
               filtered_df = df[
                   (df['Date'] < Close) &
                   (df['Strike'] == short_strike)
               ]
               result_row = None
               # Step 2: If not empty, proceed
               if not filtered_df.empty:
                   # Find the latest date
                   max_date = filtered_df['Date'].max()
                   
                   # Step 3: Filter rows having this max_date
                   max_date_rows = filtered_df[filtered_df['Date'] == max_date]
                   
                   # Step 4: Among those, pick the one with the minimum Expiration
                   result_row = max_date_rows.loc[max_date_rows['Expiration'].idxmin()]
                   stock_data= result_row
              s1= df[((df['Date'])==stock_data['Date'])  &  (df['Expiration']==stock_data['Expiration']) & (df['Strike'] == long_strike) ].copy()
              s1=s1.tail(1)
              #   print(s1,"=================")
              s_s_p=stock_data['OptionPrice'] if not stock_data.empty else 0
              l_s_p= s1['OptionPrice'].iloc[0] if not s1.empty else 0
              s_s_p=0 if s1.empty else s_s_p
              net_debit= (s_s_p - l_s_p) * contracts * 100
          
               # Get the first row for the expiration date
              stock_price = stock_data['StockPrice']
              c_d=stock_data['Date']
            portfolio_value -= net_debit
            gain= (portfolio_value-p1)/p1
            trade_chain.append({
                    'OpenDate': open_date,
                    'CloseDate': c_d,
                    'InitialPortfolioValue': round(p1,2),
                    'End_PortfolioValue': round(portfolio_value,2),
                    'Gain': round(gain*100,1),
                    'OpenPrice': trade['StockPriceAtOpen'],
                    'FinalStockPrice': stock_price,
                    'Expiration': expiration,
                    'ShortStrike': short_strike,                
                    'LongStrike': long_strike,
                    'Contracts': contracts,
                    'Buffer': round(buffer*100,2),
                    'delta': trade['ShortDelta'] ,
                    'Spread': spread,
                    'NetCredit': round(net_credit,2),
                     'NetDebit': round(net_debit,2),
                    'short_price_open': trade['ShortPrice'],
                    'long_price_open': trade['LongPrice'],
                    'short_price_close': s_s_p,
                    'long_price_close':    l_s_p,
                    'Reason': 'No condition met for rolling ',
                })
            rolled_trades_results[threshold].append(trade_chain)
            print(f"{open_date}======= {threshold}, rolling trades complete.")
        # print(f"----{count1}")
    return rolled_trades_results



import pandas as pd


def save_rolled_trades_to_csv(rolled_trades_results, filename='rolled_trades.csv'):
    all_rows = []
    for threshold, trade_chains in rolled_trades_results.items():
        COUNT=10000
        t1=10000
        for trade_id, chain in enumerate(trade_chains):
            for trade in chain:
                row = trade.copy()
                row['Short_Leg']=t1+1
                t1+=1
                row['Long_Leg']=t1+1
                t1+=1
                row['Threshold'] = threshold
                row['TradeChainID'] = trade_id
                row['Trade_ID']=COUNT+1
                COUNT+=1
                all_rows.append(row)
    
    df_all = pd.DataFrame(all_rows)

    # Ensure TradeChainID is the first column
    cols = df_all.columns.tolist()
    if 'Trade_ID' in cols:
        cols.insert(0, cols.pop(cols.index('Trade_ID')))
        df_all = df_all[cols]

    df_all.to_csv(filename, index=False)
    print(f"Saved {len(df_all)} rows to {filename}")


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

    rolled_results = simulate_rolling_trades(df, trades, thresholds=[0.1])
    save_rolled_trades_to_csv(rolled_results, 'No_alerts_0.2.csv')
    
    
if __name__ == '__main__':
    main()