import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
portfolio_value = 100000  # Initial portfolio value
# plt.style.use('seaborn-darkgrid')

# === LOAD DATA ===
file_path=r"C:\Users\rashm\OneDrive\Desktop\PRIMA2\RUT Puts 2009.xlsx"
df = pd.read_excel(file_path)

# from sqlalchemy import create_engine

# server = 'localhost'
# database = 'OptionsDataDB'
# driver = 'ODBC Driver 17 for SQL Server'

# connection_string = f"mssql+pyodbc://localhost/{database}?driver={driver.replace(' ', '+')}&trusted_connection=yes"
# engine = create_engine(connection_string)

# import pandas as pd
# df = pd.read_sql("SELECT * FROM OptionsData", engine)

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

# for y in years:
#     for m in months:
#         try:
#             d = third_friday(y, m)
#             if d in df['Expiration'].values:
#                 third_fridays.append(d)
#         except:
#             pass
# third_fridays = sorted(list(set(third_fridays)))
# print("Unique 3rd Friday expiration dates found in data:", third_fridays)
# # === 1. Identify trades: spread=20%, short strike abs(Delta) ~ 0.3, 3rd Friday expiration in following month ===

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
    count=0
    for trade_date in trade_dates:
        

        df_trade = df[(df['Date'] == trade_date) ]
        if df_trade.empty:
            continue
        
        
        candidates = df_trade[(df_trade['Delta'].abs() >= 0.28) & (df_trade['Delta'].abs() <= 0.32)]
        if candidates.empty:
            continue
        count+=1
        stock_price = df_trade['StockPrice'].iloc[0]
        candidates = candidates.copy()
        # print("Candidates for trade date", trade_date, ":", candidates)
        candidates['StrikeDiff'] = (candidates['Strike'] - stock_price).abs()
        short_strike_row = candidates.sort_values('StrikeDiff').iloc[0]
        short_strike = short_strike_row['Strike']
        
        option_price_short = short_strike_row['OptionPrice']

        long_strike = short_strike *0.85
        long_strike_row = find_nearest_strike_row(df, trade_date, short_strike_row['Expiration'] ,stock_price,long_strike)
        if long_strike_row is None:
            continue

        option_price_long = long_strike_row['OptionPrice']
        long_strike = long_strike_row['Strike']
        # print("==========================")
        # print("Short Strike Row:", short_strike_row)
        # print("Long Strike Row:", long_strike_row)
        trade = {
            'OpenDate': trade_date,
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
    # print("Number of trades found:", count)
    
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

# === 2. Define Profit/Loss at Expiration for Bull Put Spread ===


for trade in trades:
    print("=================================================================")
    print("Example trade:", trade)
   

import matplotlib.pyplot as plt




def simulate_rolling_trades(df, initial_trades, thresholds=[0.1]):
    rolled_trades_results = {th: [] for th in thresholds}
    
    for threshold in thresholds:
        count=0
        count1=0
        portfolio_value=100000  # Reset portfolio value for each threshold
        flag1=flag2=flag3=False
        trade_chain = []
        
        for trade in initial_trades:
            
                      
            flag1=False
            flag2=False
            flag3=False
            init_portfolio_value = portfolio_value  # Store initial portfolio value for this trade
            trade_chain = []
            print(f"#########Processing trade with threshold {threshold}: {trade['OpenDate']} ")
            open_date = trade['OpenDate']
            expiration = trade['Expiration']
            short_strike = trade['ShortStrike']
            long_strike = trade['LongStrike']
            SPREAD = (short_strike-long_strike)/short_strike
            net_premium1 = trade['ShortPrice'] - trade['LongPrice']
            buffer= (trade['StockPriceAtOpen']- short_strike)/ short_strike
            # Step 1: Check if stock price falls below ShortStrike before expiration
            df_initial = df[(df['Date'] > open_date) & (df['Date'] < expiration)]
            cond= df_initial['StockPrice']- short_strike < 0.025 
            contracts= float((init_portfolio_value*0.25)/(SPREAD * short_strike*100))
            contracts=round(contracts,2)
            net_credit = round(net_premium1 * contracts *100, 2)
            p1=portfolio_value
            portfolio_value += net_credit
            if cond.any():
             roll_date1 = df_initial[cond]['Date'].iloc[0]
             new_long_strike = round(0.9 * short_strike, 2)
             
            #  SPREAD= round(short_strike - new_long_strike,2)
             stock_price_on_roll = df_initial[cond]['StockPrice'].iloc[0]
             
             deb_date_row1 = df[(df['Date'] == roll_date1) & (df['Strike'] == short_strike) ].copy()
             
             d_s = deb_date_row1['OptionPrice'].iloc[0] if not deb_date_row1.empty else 0
             deb_date_row2= df[(df['Date'] == roll_date1) & (df['Expiration']==deb_date_row1['Expiration'].iloc[0])& (df['Strike'] <= long_strike) ].copy()
             deb_date_row2=deb_date_row2.loc[deb_date_row2['Strike'].idxmax()]
             d_l=  deb_date_row2['OptionPrice'] if not deb_date_row2.empty else 0
            #  print(f"=============================={d_l}")
             net_debit = d_s - d_l
            
             net_debit = round(net_debit * contracts * 100, 2)
             portfolio_value -= net_debit
             trade_chain.append({
                    'OpenDate': open_date,
                    'CloseDate': roll_date1,
                    'InitialPortfolioValue': round(p1,2),
                    'End_PortfolioValue': round(portfolio_value,2),
                    'OpenPrice': trade['StockPriceAtOpen'],
                    'FinalStockPrice': stock_price_on_roll,
                    'Expiration': expiration,
                    'ShortStrike': short_strike,
                    'LongStrike': long_strike,
                    'Contracts': contracts,
                    'Buffer': round(buffer*100,2),
                    'delta': trade['ShortDelta'],
                    'Spread': round(SPREAD*100,2),
                    'NetCredit': round(net_credit,2),
                    'NetDebit': round(net_debit,2),
                    'short_price_open': trade['ShortPrice'],
                    'long_price_open': trade['LongPrice'],
                    'short_price_close': d_s,
                    'long_price_close': d_l,
                    'Reason': 'ALert1',        
                })
             SPREAD = round((short_strike - new_long_strike)/short_strike,2)
             df_initial = df[(df['Date'] > roll_date1) & (df['Expiration'] == expiration) ].copy()
             sh= df_initial[df_initial['Strike'] == short_strike]
             
             if sh.empty :
                    print(f"Short or Long strike not found for new trade on {roll_date1}.========================== Skipping trade.")
                    
                    portfolio_value = init_portfolio_value
                    
                    trade_chain[-1]['Reason'] = 'No Short  Strike Found after alert-1'
        
                    rolled_trades_results[threshold].append(trade_chain)
                    continue
            
             lo = df[(df['Date'] == sh['Date'].iloc[0]) & (df['Strike'] <= new_long_strike) & (df['Expiration'] == expiration)].copy()

             if not lo.empty:
                 max_strike_row = lo.loc[lo['Strike'].idxmax()]
                 d_l = max_strike_row['OptionPrice']  # or max_strike_row.iloc[-1] for last element
             else:
                 portfolio_value = init_portfolio_value
                    
                 trade_chain[-1]['Reason'] = 'No  Long Strike Found after alert-1'
        
                 rolled_trades_results[threshold].append(trade_chain)
                 continue
                 
             open_date = sh['Date'].iloc[0] 
             d_s = sh['OptionPrice'].iloc[0] 
             new_premium = d_s-d_l
             new_long_price=d_l
             new_short_price=d_s
             net_credit = new_premium * contracts * 100
             net_credit = round(net_credit, 2)
             p1=portfolio_value
             portfolio_value += net_credit
             df_initial = df[(df['Date'] > open_date) & (df['Date'] < expiration)].copy()
             condition1 = df_initial['StockPrice'] <= short_strike
             o_p= sh['StockPrice'].iloc[0]
             buffer= (o_p- short_strike)/ short_strike
             if condition1.any():
                # Get first roll date
                print(f"Alert2======{open_date}")
                roll_date = df_initial[condition1]['Date'].iloc[0]
                stock_price_on_roll = df_initial[condition1]['StockPrice'].iloc[0]
               
                
                # rolled_trades_results[threshold].append(trade_chain)
                # trade_chain = []  # Reset for new trade chain
                d_r1= df_initial[(df_initial['Date'] == roll_date)   & (df_initial['Strike'] == short_strike)].copy()
                d_r2= df_initial[(df_initial['Date'] == roll_date) &(df_initial['Expiration']==d_r1['Expiration'].iloc[0]) & (df_initial['Strike'] <= new_long_strike  )  ].copy()
                
                if not d_r2.empty:
                   max_strike_row = d_r2.loc[d_r2['Strike'].idxmax()]
                   d_l = max_strike_row['OptionPrice']  # or max_strike_row.iloc[-1] for last element
                else:
                   d_l = 0
                # d_l = d_r2['OptionPrice'].iloc[0] if not d_r2.empty else 0
                d_s= d_r1['OptionPrice'].iloc[0] if not d_r1.empty else 0
                # d_l=  d_r2['OptionPrice'].iloc[0] if not d_r2.empty else 0
                net_debit = d_s - d_l
                net_debit = round(net_debit * contracts * 100, 2)
                portfolio_value -= net_debit
                trade_chain.append({
                    'OpenDate': open_date,
                    'CloseDate': roll_date,
                    'InitialPortfolioValue': round(p1,2),
                    'End_PortfolioValue': round(portfolio_value,2),
                    'OpenPrice': o_p,
                    'FinalStockPrice': stock_price_on_roll,
                    'Expiration': expiration,
                    'ShortStrike': short_strike,                
                    'LongStrike': new_long_strike,
                    'Contracts': contracts,
                    'Buffer': round(buffer*100,2),
                    'delta': sh['Delta'].iloc[0] ,
                    'Spread': round(SPREAD*100,2),
                    'NetCredit': round(net_credit,2),
                     'NetDebit': round(net_debit,2),
                    'short_price_open': new_short_price,
                    'long_price_open': new_long_price,
                    'short_price_close': d_s,
                    'long_price_close': d_l,
                    'Reason': 'Alert-2',
                })
                # New trade with same Short Strike, Long Strike = 0.9 * Short Strike, new expiration
            
                new_short_strike = short_strike  # remains same
                new_long_strike = round(0.9 * short_strike, 2)
                roll_expirations = sorted([d for d in df['Expiration'] if d >= roll_date + pd.DateOffset(months=3)])
                if not roll_expirations:
                    flag1=True
                    portfolio_value= init_portfolio_value  # Reset portfolio value if no valid expiration found
                    trade_chain.append({
                    'OpenDate': roll_date,
                    'CloseDate': '-',
                    'InitialPortfolioValue': round(p1,2),
                    'End_PortfolioValue': round(portfolio_value,2),
                    'OpenPrice': '-',
                    'FinalStockPrice': '-',
                    'Expiration': '-',
                    'ShortStrike': short_strike,                
                    'LongStrike': new_long_strike,
                    'Contracts': contracts,
                    'Buffer': '-',
                    'delta': '-' ,
                    'Spread': round(SPREAD*100,2),
                    'NetCredit': 0,
                     'NetDebit': 0,
                    'short_price_open': '-',
                    'long_price_open': '-',
                    'short_price_close': '-',
                    'long_price_close': '-',
                    'Reason': 'No expiration found after alert-2',
                })
                    rolled_trades_results[threshold].append(trade_chain)
                    continue
               
                print(f"roll_expirations found--------{roll_date},,{roll_expirations[0]}")
                # new_expiration = roll_expirations[0]

                # Always fetch fresh data from original df (not from reused df_new_trade)
                
                
                df_new_trade = df[(df['Date'] >roll_date) & (df['Expiration'] == roll_expirations[0])].copy()
                
            

                              # Add a delta filter around 0.3 for the short leg (common heuristic)
                short_row = df_new_trade[(df_new_trade['Strike'] == short_strike) ]
                if short_row.empty:
                    portfolio_value=init_portfolio_value
                    trade_chain.append({'OpenDate': roll_date,
                    'CloseDate': '-',
                    'InitialPortfolioValue': round(p1,2),
                    'End_PortfolioValue': round(portfolio_value,2),
                    'OpenPrice': '-',
                    'FinalStockPrice': '-',
                    'Expiration': roll_expirations[0],
                    'ShortStrike': short_strike,                
                    'LongStrike': new_long_strike,
                    'Contracts': contracts,
                    'Buffer': '-',
                    'delta': '-' ,
                    'Spread': round(SPREAD*100,2),
                    'NetCredit': 0,
                     'NetDebit': 0,
                    'short_price_open': '-',
                    'long_price_open': '-',
                    'short_price_close': '-',
                    'long_price_close': '-',
                    'Reason': 'No strikes found after alert-2',
                })
                    rolled_trades_results[threshold].append(trade_chain)
                    
                    print(f"Short or Long strike not found for new trade on {roll_date}. Skipping trade.")
                    continue
                    
                long_row = df_new_trade[(df_new_trade['Date']==short_row['Date'].iloc[0]) & (df_new_trade['Strike'] <= new_long_strike )  ]
                

            

                if short_row.empty or long_row.empty:
                    flag2=True
                    portfolio_value = init_portfolio_value  # Reset portfolio value if no valid strikes found
                    trade_chain.append({'OpenDate': roll_date,
                    'CloseDate': '-',
                    'InitialPortfolioValue': round(p1,2),
                    'End_PortfolioValue': round(portfolio_value,2),
                    'OpenPrice': '-',
                    'FinalStockPrice': '-',
                    'Expiration': roll_expirations[0],
                    'ShortStrike': short_strike,                
                    'LongStrike': new_long_strike,
                    'Contracts': contracts,
                    'Buffer': '-',
                    'delta': '-' ,
                    'Spread': round(SPREAD*100,2),
                    'NetCredit': 0,
                     'NetDebit': 0,
                    'short_price_open': '-',
                    'long_price_open': '-',
                    'short_price_close': '-',
                    'long_price_close': '-',
                    'Reason': 'No strikes found after alert-2',
                })
                    rolled_trades_results[threshold].append(trade_chain)
                    
                    print(f"Short or Long strike not found for new trade on {roll_date}. Skipping trade.")
                    continue
                max_strike_row=long_row.loc[long_row['Strike'].idxmax()]
                d_l1=max_strike_row['OptionPrice'] 
                new_short_price = short_row['OptionPrice'].iloc[0]
               
                net_credit = new_short_price - d_l1
                net_credit = round(net_credit * contracts * 100, 2)
                p1=portfolio_value
                portfolio_value += net_credit
                # Step 2: Track new trade for threshold rolling
                current_open = short_row['Date'].iloc[0]
                current_exp = roll_expirations[0]  # Use the first valid expiration date
                current_short = new_short_strike
                current_long = new_long_strike
                current_net = net_credit
                curr_delta= short_row['Delta'].iloc[0]
                d_s1=short_row['OptionPrice'].iloc[0]
                curr_sp= short_row['StockPrice'].iloc[0]
                
                buffer= (short_row['StockPrice'].iloc[0]- current_short)/ current_short
                l=0
                df_track = df_new_trade[(df_new_trade['Date'] > current_open) &((current_exp - df_new_trade['Date']).dt.days < 30)]
                condition = (df_track['StockPrice'] < current_short) & (df_track['StockPrice'] > current_long)
                # condition = df_track['StockPrice'] < current_short & (df_track['StockPrice'] > current_long)
                if condition.any():
                    print(f"alert3 found for threshold {threshold}, rolling trade on {current_open} with stock price {df_track['StockPrice'].iloc[0]}")
                    roll_date = df_track[condition]['Date'].iloc[0]
                    stock_price = df_track[condition]['StockPrice'].iloc[0]
                    
                    d_r1= df_track[(df_track['Date'] == roll_date) & (df_track['Strike'] >= current_short)].copy()
                    d_r2= df_track[(df_track['Date'] == roll_date) & (df_track['Expiration']==d_r1['Expiration'].iloc[0]) & (df_track['Strike']<=current_long ) ].copy()
                    max_strike_row=d_r2.loc[d_r2['Strike'].idxmax()]
                    d_l=0
                    d_s=0
                    net_debit=0
                    if not d_r1.empty and not d_r2.empty:
                      
                     d_l = max_strike_row['OptionPrice']
                     d_s= d_r1['OptionPrice'].iloc[0] 
                    # d_l=  d_r2['OptionPrice'].iloc[0] if not d_r2.empty else 0
                     net_debit = d_s - d_l
                     net_debit = round(net_debit * contracts * 100, 2)
                     portfolio_value -= net_debit
                    trade_chain.append({
                    'OpenDate': current_open,
                    'CloseDate': roll_date,
                    'InitialPortfolioValue': round(p1,2),
                    'End_PortfolioValue': round(portfolio_value,2),
                    'OpenPrice': short_row['StockPrice'].iloc[0],
                    'FinalStockPrice': stock_price,
                    'Expiration': roll_expirations[0],
                    'ShortStrike': current_short,                
                    'LongStrike': max_strike_row['Strike'] ,
                    'Contracts': contracts,
                    'Buffer': round(buffer*100,2),
                    'delta': curr_delta ,
                    'Spread': round(SPREAD*100,2),
                    'NetCredit': round(current_net,2),
                     'NetDebit': round(net_debit,2),
                    'short_price_open': d_s1,
                    'long_price_open': d_l1,
                    'short_price_close': d_s,
                    'long_price_close': d_l,
                    'Reason': 'Alert-3',
                })
                    roll_expirations = sorted([d for d in df['Expiration'] if d >= roll_expirations[0] + pd.DateOffset(months=1)])
                    current_exp=roll_expirations[0]
                    df1= df[(df['Date']>roll_date) & (df['Expiration']==roll_expirations[0])]
                    if df1.empty:
                      portfolio_value = init_portfolio_value  # Reset portfolio value if no valid expiration found
                #   for t in trade_chain:
                      trade_chain[-1]['Reason'] = 'No Expiartion or open_date Found'
                    #   trade_chain[-1]['PortfolioValue'] = init_portfolio_value
                      rolled_trades_results[threshold].append(trade_chain) 
                      continue
                    short_r1= df1[(df1['Strike'] == current_short) ].copy()
                    if short_r1.empty:
                        portfolio_value = init_portfolio_value  # Reset portfolio value if no valid expiration found
                        trade_chain[-1]['Reason'] = 'No  short strikes found after alert3'
                        # trade_chain[-1]['PortfolioValue'] = init_portfolio_value
                        rolled_trades_results[threshold].append(trade_chain) 
                        continue
                        
                    long_r1=df1[(df1['Date']==short_r1['Date'].iloc[0]) & (df1['Strike']<=current_long) ].copy()
                    long_r1= long_r1.loc[long_r1['Strike'].idxmax()]
                    
                    if short_r1.empty or long_r1.empty:
                        portfolio_value = init_portfolio_value  # Reset portfolio value if no valid expiration found
                        trade_chain[-1]['Reason'] = 'No long strike found after alert3'
                        # trade_chain[-1]['PortfolioValue'] = init_portfolio_value
                        rolled_trades_results[threshold].append(trade_chain) 
                        continue
                        
                    if not short_r1.empty and not long_r1.empty:
                      current_open=short_r1['Date'].iloc[0]
                      curr_delta= short_r1['Delta'].iloc[0]
                      curr_sp= short_r1['StockPrice'].iloc[0]
                      buffer= (short_r1['StockPrice'].iloc[0]- current_short)/ current_short
                      net_credit= short_r1['OptionPrice'].iloc[0] - long_r1['OptionPrice']
                      d_s1= short_r1['OptionPrice'].iloc[0]
                      d_l1=long_r1['OptionPrice']
                      net_credit = round(net_credit * contracts * 100, 2)
                      new_short_price= short_r1['OptionPrice'].iloc[0]
                      p1=portfolio_value
                      portfolio_value += net_credit
                      
                    
                    # new_net_premium2= short_r1['OptionPrice'].iloc[0] - long_r1['OptionPrice'].iloc[0]
                print(f"alert3 not found")
            
                while True:
                    df_track = df[(df['Date'] > current_open) & (df['Date'] < current_exp)]
                    condition2 = ((current_long*0.9)>=df_track['StockPrice'] ) 
                    if not condition2.any():
                          # Reset portfolio value if no condition met
                        
                        if df_track.empty:
                              print(f"No data to track between {current_open} and {current_exp}. Skipping final PnL calc.")
                              break
                        ex=df[(df['Date'] == current_exp ) & (df['Strike']<=current_short) ]
                        net_debit=0
                        s_s_p=0
                        l_s_p=0
                        if not ex.empty:
                            c_d=current_exp
                            stock_data=ex
                            f_p=ex['StockPrice'].iloc[0]
                        if  ex.empty:
                              # break  # or continue, depending on context
                          filtered_df = df[
                           (df['Date'] <= current_exp) &
                           (df['Strike'] >= current_short)
                            ]
                          stock_data = None
                          # Step 2: If not empty, proceed
                          if not filtered_df.empty:
                              # Find the latest date
                              max_date = filtered_df['Date'].max()
                              
                              # Step 3: Filter rows having this max_date
                              max_date_rows = filtered_df[filtered_df['Date'] == max_date]
                              
                              # Step 4: Among those, pick the one with the minimum Expiration
                              result_row = max_date_rows.loc[max_date_rows['Expiration'].idxmin()]
                              stock_data= result_row
                              s1= df[((df['Date'])==stock_data['Date'])  &  (df['Expiration']==stock_data['Expiration']) & (df['Strike'] <= current_long) ].copy()
                              print(f"=------------{stock_data['Date']},,,,,{stock_data['Expiration']} ")
                              s1= s1.loc[s1['Strike'].idxmax()]
                              
               #                print(s1,"=================")
                              s_s_p=stock_data['OptionPrice'] if not stock_data.empty else 0
                              l_s_p= s1['OptionPrice'] if not s1.empty else 0
                              s_s_p=0 if s1.empty else s_s_p
                              net_debit= (s_s_p - l_s_p) * contracts * 100
                #             Get the first row for the expiration date
                              f_p = stock_data['StockPrice']
                              c_d=stock_data['Date']
                              
                        
                        trade_chain.append({
                         'OpenDate': current_open,
                         'CloseDate': c_d,
                         'InitialPortfolioValue': round(p1,2),
                         'End_PortfolioValue': round(portfolio_value,2),
                         'OpenPrice': curr_sp,
                         'FinalStockPrice': f_p,
                         'Expiration': current_exp,
                         'ShortStrike': current_short,                
                         'LongStrike': current_long,
                         'Contracts': contracts,
                         'Buffer': round(buffer*100,2),
                         'delta': curr_delta ,
                         'Spread': round(SPREAD*100,2),
                         'NetCredit': round(net_credit,2),
                          'NetDebit':round(net_debit,2),
                         'short_price_open': d_s1,
                         'long_price_open': d_l1,
                         'short_price_close': s_s_p,
                         'long_price_close':     l_s_p,
                         'Reason': 'No  alert-4',
                         })
                       
                        break
                    count=1
                    l=1
                    count1=count1+1
                    print(f"condition2 found for threshold {threshold}, rolling trade on {current_open} with stock price {df_track['StockPrice'].iloc[0]}")
                    roll_date2 = df_track[condition2]['Date'].iloc[0]
                    stock_price2 = df_track[condition2]['StockPrice'].iloc[0]
                  
                    d_r1= df_track[(df_track['Date'] == roll_date2) & (df_track['Strike']>=current_short )].copy()
                    if d_r1.empty:
                        portfolio_value=init_portfolio_value
                        flag2=True
                        break
                    d_r2= df_track[(df_track['Date'] == roll_date2) & (df_track['Expiration']==d_r1['Expiration'].iloc[0])&(df_track['Strike']<=current_long)  ].copy()
                    d_r2= d_r2.loc[d_r2['Strike'].idxmax()]
                    
                    d_s= d_r1['OptionPrice'].iloc[0] if not d_r1.empty else 0
                    d_l=  d_r2['OptionPrice'] if not d_r2.empty else 0
                    
                    net_debit = d_s - d_l
                    net_debit = round(net_debit * contracts * 100, 2)
                    p1=portfolio_value
                    portfolio_value -= net_debit
                    trade_chain.append({
                    'OpenDate': current_open,
                    'CloseDate': roll_date2,
                    'InitialPortfolioValue': round(p1,2),
                    'End_PortfolioValue': round(portfolio_value,2),
                    'OpenPrice': curr_sp,
                    'FinalStockPrice': stock_price2,
                    'Expiration': roll_expirations[0],
                    'ShortStrike': current_short,                
                    'LongStrike': current_long,
                    'Contracts': contracts,
                    'Buffer': round(buffer*100,2),
                    'delta': curr_delta ,
                    'Spread': round(SPREAD*100,2),
                    'NetCredit': round(net_credit,2),
                     'NetDebit': round(net_debit,2),
                    'short_price_open': d_s1,
                    'long_price_open': d_l1,
                    'short_price_close': d_s,
                    'long_price_close':     d_l,
                    'Reason': 'Alert-4',
                })

                    # if method == 'method1':
                    new_long = round((1 + 0.5 * threshold) * stock_price2, 2)
                    
                    new_short = round(new_long /(1-SPREAD), 2)

                    df_new_trade2 = df[(df['Date'] > roll_date2) & (df['Expiration'] == current_exp)].copy()
                    

                    short_row2 = df_new_trade2[(df_new_trade2['Strike'] >= new_short) ]
                    
                    

                    if not short_row2.empty :
                        portfolio_value =init_portfolio_value
                        flag3=True
                        print(f"Short or Long strike not found for new trade on {roll_date2}.{new_long}======={new_short} Skipping trade--2.")
                        break
                    long_row2 = df_new_trade2[ (df_new_trade2['Date']==short_row2['Date'].iloc[0])&(df_new_trade2['Strike'] <=new_long )  ]
                    long_row2=long_row2.loc[long_row2['Strike'].idxmax()]
                    
                    # SPREAD = short_row2['Strike'].iloc[0] - long_row2['Strike'].iloc[0]
                    d_l1 = long_row2['OptionPrice']
                    d_s1 = short_row2['OptionPrice'].iloc[0]
                    
                    new_net = d_s1 - d_l1
                    curr_delta = short_row2['Delta']
                    buffer = (short_row2['StockPrice'].iloc[0] - new_short) / new_short
                    curr_sp = short_row2['StockPrice'].iloc[0]
                    net_credit = new_net * contracts * 100
                    net_credit = round(net_credit, 2)
                    p1=portfolio_value
                    portfolio_value += net_credit
                    current_open = df_new_trade2['Date'].iloc[0]
                    current_short = new_short
                    current_long = new_long
                    

                if not flag1 and not flag2 and not flag3:
                 rolled_trades_results[threshold].append(trade_chain)
                if flag2:
                     portfolio_value = init_portfolio_value  # Reset portfolio value if no valid expiration found
                #   for t in trade_chain:
                     trade_chain[-1]['Reason'] = 'No strikes at debit'
                     trade_chain[-1]['PortfolioValue'] = init_portfolio_value
                     rolled_trades_results[threshold].append(trade_chain) 
                elif flag3 :
                    portfolio_value = init_portfolio_value  # Reset portfolio value if no valid strikes found
                    # for t in trade_chain:
                    trade_chain[-1]['Reason'] = 'No Short or Long Strike Found after alert-4'
                    # trade_chain[-1]['PortfolioValue'] = init_portfolio_value
                    rolled_trades_results[threshold].append(trade_chain) 
             else:
              
                   # Step 1: Filter rows where Date < expiration_date and Strike == short_strike
                  filtered_df = df[
                      (df['Date'] <= expiration) &
                      (df['Strike'] == short_strike)
                  ]
                  stock_data = None
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
                  s1= df[((df['Date'])==stock_data['Date'])  &  (df['Expiration']==stock_data['Expiration']) & (df['Strike'] == new_long_strike) ].copy()
                  s1=s1.tail(1)
               #    print(s1,"=================")
                  s_s_p=stock_data['OptionPrice'] if not stock_data.empty else 0
                  l_s_p= s1['OptionPrice'].iloc[0] if not s1.empty else 0
                  s_s_p=0 if s1.empty else s_s_p
                  net_debit= (s_s_p - l_s_p) * contracts * 100
           
                # Get the first row for the expiration date
                  stock_price = stock_data['StockPrice']
                  c_d=stock_data['Date']
               
               
                  portfolio_value -= net_debit
                  trade_chain.append({
                    'OpenDate': open_date,
                    'CloseDate': c_d,
                    'InitialPortfolioValue': round(p1,2),
                    'End_PortfolioValue': round(portfolio_value,2),
                    'OpenPrice': o_p,
                    'FinalStockPrice': stock_price,
                    'Expiration': expiration,
                    'ShortStrike': short_strike,                
                    'LongStrike': new_long_strike,
                    'Contracts': contracts,
                    'Buffer': round(buffer*100,2),
                    'delta': sh['Delta'].iloc[0] ,
                     'Spread': round(SPREAD*100,2),
                    'NetCredit': round(net_credit,2),
                     'NetDebit': round(net_debit,2),
                    'short_price_open': new_short_price,
                    'long_price_open': new_long_price,
                    'short_price_close': s_s_p,
                    'long_price_close': l_s_p,
                    'Reason': 'No condition met for rolling afetr alert-1',
                })
                  rolled_trades_results[threshold].append(trade_chain)
            else:
             expiration_date = trade['Expiration']
             
             stock_data = df[(df['Date'] == expiration_date) & (df['Strike'] == short_strike) ]
             if not stock_data.empty:
              stock_data= stock_data.iloc[0]
             net_debit=0
             s_s_p=0
             l_s_p=0
             
             if stock_data.empty:
                 # Step 1: Filter rows where Date < expiration_date and Strike == short_strike
                filtered_df = df[
                    (df['Date'] < expiration_date) &
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
             #    print(s1,"=================")
                s_s_p=float(stock_data['OptionPrice']) if not stock_data.empty else 0
                l_s_p= s1['OptionPrice'].iloc[0] if not s1.empty else 0
                s_s_p=0 if s1.empty else s_s_p
                net_debit= (s_s_p - l_s_p) * contracts * 100
         
              # Get the first row for the expiration date
             stock_price = stock_data['StockPrice']
             c_d=stock_data['Date']
             
             
             portfolio_value -= net_debit
               
             trade_chain.append({
                     'OpenDate': open_date,
                     'CloseDate': c_d,
                     'InitialPortfolioValue': round(p1,2),
                     'End_PortfolioValue': round(portfolio_value,2),
                     'OpenPrice': trade['StockPriceAtOpen'],
                     'FinalStockPrice': stock_price,
                     'Expiration': expiration,
                     'ShortStrike': short_strike,                
                     'LongStrike': long_strike,
                     'Contracts': contracts,
                     'Buffer': round(buffer*100,2),
                     'delta': trade['ShortDelta'] ,
                     'Spread':round(SPREAD*100,2),
                     'NetCredit': round(net_credit,2),
                      'NetDebit': round(net_debit,2),
                     'short_price_open': trade['ShortPrice'],
                     'long_price_open': trade['LongPrice'],
                     'short_price_close': s_s_p,
                     'long_price_close':    l_s_p,
                     'Reason': 'No condition met for rolling ',
                 })
             rolled_trades_results[threshold].append(trade_chain)
             print(f"condition1 not found for threshold {threshold}, rolling trades complete.")
        print(f"----{count1}")
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
    save_rolled_trades_to_csv(rolled_results, 'rolled_trades_output_2009.csv')
    
    
if __name__ == '__main__':
    main()