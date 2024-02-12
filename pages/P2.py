import streamlit as st
import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
import bokeh
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report, balanced_accuracy_score

# Import BalancedRandomForestClassifier from imblearn
from imblearn.ensemble import BalancedRandomForestClassifier

# Import Support Vector Machine from sklearn
from sklearn import svm

# Import AdaBoostClassifier from sklearn
from sklearn.ensemble import AdaBoostClassifier

# Import the finta Python library and the TA module
from finta import TA


def main():
    st.title("Individual Stock Analysis")

    start = st.date_input("Enter the Starting Point for the Historical Data (Preferable 5 years or Less)", value=dt.datetime(2020, 1, 1), max_value=dt.datetime.now(), format="DD.MM.YYYY")

    # Calculate start and end dates
    end_date = dt.datetime.now()
    start_date = start

    st.write(f"Start Date: {start_date}")
    st.write(f"End Date: {end_date.strftime('%Y-%m-%d')}")

    # Multi-select option for stock tickers
    df = pd.read_csv('tickers.csv')
    selected_stock = st.selectbox("Please Select a Stock", options=df.Symbol)

    # Fetch historical price data for the selected stocks using yfinance
    try:
        stock = yf.download(selected_stock, start=start_date, end=end_date)

        # Extract the closing prices for each selected stock
        stock = pd.DataFrame(stock['Close'])

        # Ensure the index is in datetime format
        stock.index = pd.to_datetime(stock.index, format='mixed', infer_datetime_format=True)
        
        # Plot daily returns of the portfolio
        st.subheader("Historical Closing Price")
        st.line_chart(stock)

        # Option to choose trading algorithm
        trading_algorithm = st.selectbox("Choose a Trading Algorithm", options=["Dual-Moving Average Crossover (DMAC)", "Bollinger Bands", "RSI Oscillator"])

        if trading_algorithm == "Dual-Moving Average Crossover (DMAC)":
            st.subheader("Dual-Moving Average Crossover (DMAC)")
            
            st.write("The Dual-Moving Average Crossover Trading Algorithm is a strategy that generates buy or sell signals based on the intersection of two different moving averages, typically a shorter-term average crossing above or below a longer-term average, indicating potential shifts in market trends.")

            st.write("The moving averages in this algorithm are taken as the exponential moving average (EMA) which is an exponentially weighted average of the previous 'n' closing periods. More recent closing prices are weighted heavier than older closing periods providing an average that is faster to respond to changing prices.")

            st.write("A simple strategy is applied to generate the trade signal where `1` (representing a buy signal) is generated whenever the short window moving average is greater than the long window. A `0` (representing a sell signal) is generated when the opposite it the case. The entry/exit points are thus created and the variables visualised.")

            # Option to select short and long window
            short_window = st.number_input("Enter Short Window", min_value=1, max_value=len(stock)//2, step=1, value=10)
            long_window = st.number_input("Enter Long Window", min_value=1, max_value=len(stock)//2, step=1, value=50)

            # Generate the fast and slow exponentially moving averages
            stock['EMA_Fast'] = stock['Close'].ewm(span=short_window).mean()
            stock['EMA_Slow'] = stock['Close'].ewm(span=long_window).mean()

            # Drop NaN Values
            stock = stock.dropna()

            # Create a column to hold the trading signal
            stock["Signal"] = 0.0

            # Generate the trading signal 0 or 1, where 1 is the short-window (EMA_Fast) greater than the long-window (EMA_Slow) and 0 is when the condition is not met
            stock["Signal"][short_window:] = np.where(
                stock["EMA_Fast"][short_window:] > stock["EMA_Slow"][short_window:], 1.0, 0.0
            )

            # Calculate the points in time when the Signal value changes
            # Identify trade entry (1) and exit (-1) points
            stock["Entry/Exit"] = stock["Signal"].diff()
            
            # Create traces
            trace_close = go.Scatter(x=stock.index, y=stock['Close'], mode='lines', name='Close')
            trace_ema_fast = go.Scatter(x=stock.index, y=stock['EMA_Fast'], mode='lines', name='EMA Fast')
            trace_ema_slow = go.Scatter(x=stock.index, y=stock['EMA_Slow'], mode='lines', name='EMA Slow')
            trace_entry = go.Scatter(x=stock[stock["Entry/Exit"] == 1.0].index, y=stock[stock["Entry/Exit"] == 1.0]['Close'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy')
            trace_exit = go.Scatter(x=stock[stock["Entry/Exit"] == -1.0].index, y=stock[stock["Entry/Exit"] == -1.0]['Close'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell')

            # Create figures
            fig = go.Figure()
            fig2 = go.Figure()

            # Add traces
            fig.add_trace(trace_close)
            fig.add_trace(trace_entry)
            fig.add_trace(trace_exit)
            
            fig2.add_trace(trace_close)
            fig2.add_trace(trace_ema_fast)
            fig2.add_trace(trace_ema_slow)

            # Show plot
            st.plotly_chart(fig2)
            
            # Title
            st.subheader(f"{selected_stock} - Closing Price, Entry and Exit Points")

            # Show plot
            st.plotly_chart(fig)
            
            # Backtesting
            st.subheader("Backtesting")
            
            # Option to select initial
            initial_capital = st.number_input("Enter Initial Capital", min_value=1, step=1, value=10000)
            share_size = st.number_input("Enter Share Size", min_value=1, step=1, value=100)
            
            # Buy the chosen share position when the dual moving average crossover Signal equals 1
            # Otherwise, `Position` should be zero (sell)
            stock['Position'] = share_size * stock['Signal']
            
            # Determine the points in time where a 500 share position is bought or sold
            stock['Entry/Exit Position'] = stock['Position'].diff()
            
            # Multiply the close price by the number of shares held, or the Position
            stock['Portfolio Holdings'] = stock['Close'] * stock['Position']
            
            # Subtract the amount of either the cost or proceeds of the trade from the initial capital invested
            stock['Portfolio Cash'] = initial_capital - (stock['Close'] * stock['Entry/Exit Position']).cumsum() 
            
            # Calculate the total portfolio value by adding the portfolio cash to the portfolio holdings (or investments)
            stock['Portfolio Total'] = stock['Portfolio Cash'] + stock['Portfolio Holdings']
            
            # Calculate the portfolio daily returns
            stock['Portfolio Daily Returns'] = stock['Portfolio Total'].pct_change()
            
            # Calculate the portfolio cumulative returns
            stock['Portfolio Cumulative Returns'] = (1 + stock['Portfolio Daily Returns']).cumprod() - 1
            
            # Create traces
            total_portfolio_trace = go.Scatter(x=stock.index, y=stock['Portfolio Total'], mode='lines', name='Total Portfolio Value')
            
            exit_position_trace = go.Scatter(x=stock[stock['Entry/Exit'] == -1.0].index, y=stock[stock['Entry/Exit'] == -1.0]['Portfolio Total'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell')

            entry_position_trace = go.Scatter(x=stock[stock['Entry/Exit'] == 1.0].index, y=stock[stock['Entry/Exit'] == 1.0]['Portfolio Total'], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy')
            

            # Create figure
            fig3 = go.Figure()

            # Add traces
            fig3.add_trace(total_portfolio_trace)
            fig3.add_trace(exit_position_trace)
            fig3.add_trace(entry_position_trace)

            # Show plot
            st.plotly_chart(fig3)
            
            
            # Risk Reward Characteristics
            st.subheader("Risk-Reward Characteristics")
            
            # Calculate Annualized Returns
            annualized_returns = stock["Portfolio Daily Returns"].mean() * 252
            st.write(f"**Annualized Returns**: {annualized_returns}")
            
            # Calculate Cumulative Returns
            cumulative_returns = stock["Portfolio Cumulative Returns"][-1]
            st.write(f"**Cumulative Returns**: {cumulative_returns}")
            
            # Calculate Annual Volatility
            annual_volatility = stock["Portfolio Daily Returns"].std() * np.sqrt(252)
            st.write(f"**Annual Volatility**: {annual_volatility}")
            
            # Calculate Sharpe ratio
            sharpe_ratio = (stock["Portfolio Daily Returns"].mean() * 252) / (stock["Portfolio Daily Returns"].std() * np.sqrt(252))
            st.write(f"**Sharpe Ratio**: {sharpe_ratio}")
            
            
            # Machine Learning Implementation
            st.subheader("Machine Learning Model")
            
            st.write("The data is split into training and testing components using date offsets to create rolling windows (75-25 split). It is then standardised using `StandardScalar` function.")
            
            # Use the pct_change function to generate daily returns from the closing prices
            stock["Daily Returns"] = stock['Close'].pct_change()
            
            # Calculate the strategy returns and add them to the stock DataFrame
            stock['Strategy Returns'] = stock['Daily Returns'] * stock['Signal'].shift()
            
            # Assign a copy of the ema_fast and ema_slow columns to a features DataFrame called X1
            X1 = stock[['EMA_Slow', 'EMA_Fast']].shift().dropna()
            
            # Create the target set selecting the Signal column and assiging it to y1
            y1 = stock['Signal']
            
            
            # Date Offset to create train-test split
            train_duration = 3  # months
            step_size = 1  # month

            # Initialize lists to store training and testing dataframes
            all_X1_train = []
            all_y1_train = []
            all_X1_test = []
            all_y1_test = []

            # Initialize the starting point for the training period
            current_start = X1.index.min()

            # Loop until the end of the dataset
            while current_start + pd.DateOffset(months=train_duration) <= X1.index.max():
                # Define the training end period
                training_end = current_start + pd.DateOffset(months=train_duration)

                # Generate the training data for the current window
                X1_train = X1.loc[current_start:training_end]
                y1_train = y1.loc[current_start:training_end]
                all_X1_train.append(X1_train)
                all_y1_train.append(y1_train)

                # Define the testing period as the next time step after the training end
                testing_end = training_end + pd.DateOffset(months=step_size)
                if testing_end > X1.index.max():
                    testing_end = X1.index.max()

                # Generate the testing data for the current window
                X1_test = X1.loc[training_end:testing_end]
                y1_test = y1.loc[training_end:testing_end]
                all_X1_test.append(X1_test)
                all_y1_test.append(y1_test)

                # Update the start for the next iteration
                current_start = testing_end

            # Concatenate all training and testing dataframes
            final_X1_train = pd.concat(all_X1_train)
            final_y1_train = pd.concat(all_y1_train)
            final_X1_test = pd.concat(all_X1_test)
            final_y1_test = pd.concat(all_y1_test)
            
            # Scale the features DataFrames

            # Create a StandardScaler instance
            scaler1 = StandardScaler()

            # Apply the scaler model to fit the X1_train data
            X1_scaler = scaler1.fit(final_X1_train)

            # Transform the X_train and X_test DataFrames using the X1_scaler
            X1_train_scaled = X1_scaler.transform(final_X1_train)
            X1_test_scaled = X1_scaler.transform(final_X1_test)
            
            # Option to choose machine learning model
            ml_model = st.selectbox("Choose a Machine Learning Model", options=["Balanced Random Forest", "Support Vector Machines", "AdaBoost Classifier"])

            if ml_model == "Support Vector Machines":
                
                # Create the support vector classifier model
                svm_model = svm.SVC()

                # Fit the model to the training data
                svm_model.fit(X1_train_scaled, final_y1_train)
                
                # Predict labels for testing features
                y1_prediction_svm = svm_model.predict(X1_test_scaled)
                
                # Generate classification report
                report = classification_report(final_y1_test, y1_prediction_svm, output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                # Display classification report in Streamlit
                st.write(report_df)
            
            elif ml_model == "Balanced Random Forest":
                
                # Instantiate a BalancedRandomForestClassifier instance
                brf = BalancedRandomForestClassifier(sampling_strategy='all', replacement=True)

                # Fit the model to the training data
                brf.fit(X1_train_scaled, final_y1_train)
                
                # Predict labels for testing features
                y1_prediction_brf = brf.predict(X1_test_scaled)
                
                # Generate classification report
                report = classification_report(final_y1_test, y1_prediction_brf, output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                # Display classification report in Streamlit
                st.write(report_df)
            
            elif ml_model == "AdaBoost Classifier":
                
                # Create the ada boost classifier
                adaboost = AdaBoostClassifier(n_estimators=50, random_state=0)
                
                # Fit the model to the training data
                adaboost.fit(X1_train_scaled, final_y1_train)
                
                # Predict labels for testing features
                y1_prediction_ada = adaboost.predict(X1_test_scaled)
                
                # Generate classification report
                report = classification_report(final_y1_test, y1_prediction_ada, output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                # Display classification report in Streamlit
                st.write(report_df)
                
    
        elif trading_algorithm == "Bollinger Bands":
            st.subheader("Bollinger Bands")
            
            st.write("The Bollinger Bands consist of three lines: a simple moving average (SMA) line at the centre and two additional lines above and below the SMA that represent the standard deviations (usually two) of price movements giving it a dynamic nature and versatility.")

            st.write("The trading signals examine the closing price against the upper and lower Bollinger Band thresholds. When the closing price falls below the lower band (`BB_LOWER`), it triggers a buy signal (`Signal = 1`) only if no existing buy signal is active (`trade_signal < 1`). Subsequently, it tracks the buy price.")

            st.write("To manage risk, it sets a stop-loss condition (`here 30%`). If the stock's price drops below a specified stop-loss percentage from the buy price, it triggers a sell signal (`Signal = 0`) and resets the buy signal (`trade_signal = 0`). Moreover, it generates a sell signal when the price rises above the upper band (`BB_UPPER`) after a buy signal (`trade_signal > 0`). The algorithm is designed to ensure only one entry and exit point per trade cycle, maintaining or closing the position based on these conditions while monitoring trade signals.")
            
            # Fetch historical price data for the specified stock using yfinance
            stock_df = yf.download(selected_stock, start=start_date, end=end_date)

            # Extract the closing prices for each stock
            stock_df = pd.DataFrame(stock_df)

            # Determine the Bollinger Bands for the Dataset
            bbands_df = TA.BBANDS(stock_df)

            # Concatenate the Bollinger Bands to the DataFrame
            stock_df = pd.concat([stock_df, bbands_df], axis=1)

            # Option to select stop loss percentage
            stop_percentage = st.number_input("Enter Stop Loss Percentage", min_value=0, max_value=100, step=1, value=10)

            # Create a trading algorithm using Bollinger Bands
            # Set the Signal column
            stock_df["Signal"] = 0.0

            # Create a value to hold the initial trade signal and the buy price
            trade_signal = 0
            buy_price = 0.0

            # Set the stop loss percentage (e.g., 10%)
            stop_loss_percentage = stop_percentage / 100

            # Update the DataFrame Signal column 1 (entry) or -1 (exit) for a long position trading algorithm
            # where 1 is when the Close price is less than the BB_LOWER window
            # where -1 is when the Close price is greater the the BB_UPPER window
            # Incorporate a conditional in the if-statement, to evaluate the value of the trade_signal so the algorithm 
            # plots only 1 entry and exit point per cycle.
            for index, row in stock_df.iterrows():
                # Entry condition
                if (row["Close"] < row["BB_LOWER"]) and (trade_signal < 1):
                    stock_df.loc[index, "Signal"] = 1.0
                    trade_signal += 1
                    buy_price = row["Close"]  # Record the buy price

                # Stop loss condition
                elif (buy_price > 0) and (row["Close"] < buy_price * (1 - stop_loss_percentage)) and (trade_signal > 0):
                    stock_df.loc[index, "Signal"] = 0
                    trade_signal = 0
                    buy_price = 0.0  # Reset the buy price

                # Exit condition
                elif (row["Close"] > row["BB_UPPER"]) and (trade_signal > 0):
                    stock_df.loc[index, "Signal"] = 0
                    trade_signal = 0
                    buy_price = 0.0  # Reset the buy price

                # Maintain position
                elif trade_signal == 1:
                    stock_df.loc[index, "Signal"] = 1.0  # Maintain the position

            # Calculate the points in time when the Signal value changes
            # Identify trade entry (1) and exit (-1) points
            stock_df["Entry/Exit"] = stock_df["Signal"].diff()

            # Create traces
            trace_close = go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name='Close')
            trace_bb_middle = go.Scatter(x=stock_df.index, y=stock_df['BB_MIDDLE'], mode='lines', name='BB Middle')
            trace_bb_upper = go.Scatter(x=stock_df.index, y=stock_df['BB_UPPER'], mode='lines', name='BB Upper')
            trace_bb_low = go.Scatter(x=stock_df.index, y=stock_df['BB_LOWER'], mode='lines', name='BB Lower')
            trace_entry = go.Scatter(x=stock_df[stock_df["Entry/Exit"] == 1.0].index, y=stock_df[stock_df["Entry/Exit"] == 1.0]['Close'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy')
            trace_exit = go.Scatter(x=stock_df[stock_df["Entry/Exit"] == -1.0].index, y=stock_df[stock_df["Entry/Exit"] == -1.0]['Close'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell')

            # Create figures
            fig = go.Figure()
            fig2 = go.Figure()

            # Add traces
            fig.add_trace(trace_close)
            fig.add_trace(trace_entry)
            fig.add_trace(trace_exit)

            fig2.add_trace(trace_close)
            fig2.add_trace(trace_bb_middle)
            fig2.add_trace(trace_bb_upper)
            fig2.add_trace(trace_bb_low)

            # Show plot
            st.plotly_chart(fig2)

            # Title
            st.subheader(f"{selected_stock} - Closing Price, Entry and Exit Points")

            # Show plot
            st.plotly_chart(fig)
            
            # Backtesting
            st.subheader("Backtesting")

            # Option to select initial
            initial_capital = st.number_input("Enter Initial Capital", min_value=1, step=1, value=10000)
            share_size = st.number_input("Enter Share Size", min_value=1, step=1, value=100)

            # Buy the chosen share position when the dual moving average crossover Signal equals 1
            # Otherwise, `Position` should be zero (sell)
            stock_df['Position'] = share_size * stock_df['Signal']

            # Determine the points in time where a 500 share position is bought or sold
            stock_df['Entry/Exit Position'] = stock_df['Position'].diff()

            # Multiply the close price by the number of shares held, or the Position
            stock_df['Portfolio Holdings'] = stock_df['Close'] * stock_df['Position']

            # Subtract the amount of either the cost or proceeds of the trade from the initial capital invested
            stock_df['Portfolio Cash'] = initial_capital - (stock_df['Close'] * stock_df['Entry/Exit Position']).cumsum() 

            # Calculate the total portfolio value by adding the portfolio cash to the portfolio holdings (or investments)
            stock_df['Portfolio Total'] = stock_df['Portfolio Cash'] + stock_df['Portfolio Holdings']

            # Calculate the portfolio daily returns
            stock_df['Portfolio Daily Returns'] = stock_df['Portfolio Total'].pct_change()

            # Calculate the portfolio cumulative returns
            stock_df['Portfolio Cumulative Returns'] = (1 + stock_df['Portfolio Daily Returns']).cumprod() - 1

            # Create traces
            total_portfolio_trace = go.Scatter(x=stock_df.index, y=stock_df['Portfolio Total'], mode='lines', name='Total Portfolio Value')

            exit_position_trace = go.Scatter(x=stock_df[stock_df['Entry/Exit'] == -1.0].index, y=stock_df[stock_df['Entry/Exit'] == -1.0]['Portfolio Total'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell')

            entry_position_trace = go.Scatter(x=stock_df[stock_df['Entry/Exit'] == 1.0].index, y=stock_df[stock_df['Entry/Exit'] == 1.0]['Portfolio Total'], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy')

            # Create figure
            fig3 = go.Figure()

            # Add traces
            fig3.add_trace(total_portfolio_trace)
            fig3.add_trace(exit_position_trace)
            fig3.add_trace(entry_position_trace)

            # Show plot
            st.plotly_chart(fig3)


            # Risk Reward Characteristics
            st.subheader("Risk-Reward Characteristics")

            # Calculate Annualized Returns
            annualized_returns = stock_df["Portfolio Daily Returns"].mean() * 252
            st.write(f"**Annualized Returns**: {annualized_returns}")

            # Calculate Cumulative Returns
            cumulative_returns = stock_df["Portfolio Cumulative Returns"][-1]
            st.write(f"**Cumulative Returns**: {cumulative_returns}")

            # Calculate Annual Volatility
            annual_volatility = stock_df["Portfolio Daily Returns"].std() * np.sqrt(252)
            st.write(f"**Annual Volatility**: {annual_volatility}")

            # Calculate Sharpe ratio
            sharpe_ratio = (stock_df["Portfolio Daily Returns"].mean() * 252) / (stock_df["Portfolio Daily Returns"].std() * np.sqrt(252))
            st.write(f"**Sharpe Ratio**: {sharpe_ratio}")


            # Machine Learning Implementation
            st.subheader("Machine Learning Model")

            st.write("The data is split into training and testing components using date offsets to create rolling windows (75-25 split). It is then standardised using `StandardScalar` function.")

            # Use the pct_change function to generate daily returns from the closing prices
            stock_df["Daily Returns"] = stock_df['Close'].pct_change()

            # Calculate the strategy returns and add them to the stock DataFrame
            stock_df['Strategy Returns'] = stock_df['Daily Returns'] * stock_df['Signal'].shift()

            # Assign a copy of the ema_fast and ema_slow columns to a features DataFrame called X1
            X1 = stock_df[['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']].shift().dropna()

            # Create the target set selecting the Signal column and assiging it to y1
            y1 = stock_df['Signal']

            # Date Offset to create train-test split
            train_duration = 3  # months
            step_size = 1  # month

            # Initialize lists to store training and testing dataframes
            all_X1_train = []
            all_y1_train = []
            all_X1_test = []
            all_y1_test = []

            # Initialize the starting point for the training period
            current_start = X1.index.min()

            # Loop until the end of the dataset
            while current_start + pd.DateOffset(months=train_duration) <= X1.index.max():
                # Define the training end period
                training_end = current_start + pd.DateOffset(months=train_duration)

                # Generate the training data for the current window
                X1_train = X1.loc[current_start:training_end]
                y1_train = y1.loc[current_start:training_end]
                all_X1_train.append(X1_train)
                all_y1_train.append(y1_train)

                # Define the testing period as the next time step after the training end
                testing_end = training_end + pd.DateOffset(months=step_size)
                if testing_end > X1.index.max():
                    testing_end = X1.index.max()

                # Generate the testing data for the current window
                X1_test = X1.loc[training_end:testing_end]
                y1_test = y1.loc[training_end:testing_end]
                all_X1_test.append(X1_test)
                all_y1_test.append(y1_test)

                # Update the start for the next iteration
                current_start = testing_end

            # Concatenate all training and testing dataframes
            final_X1_train = pd.concat(all_X1_train)
            final_y1_train = pd.concat(all_y1_train)
            final_X1_test = pd.concat(all_X1_test)
            final_y1_test = pd.concat(all_y1_test)

            # Scale the features DataFrames

            # Create a StandardScaler instance
            scaler1 = StandardScaler()

            # Apply the scaler model to fit the X1_train data
            X1_scaler = scaler1.fit(final_X1_train)

            # Transform the X_train and X_test DataFrames using the X1_scaler
            X1_train_scaled = X1_scaler.transform(final_X1_train)
            X1_test_scaled = X1_scaler.transform(final_X1_test)
            
            # Option to choose machine learning model
            ml_model = st.selectbox("Choose a Machine Learning Model", options=["Balanced Random Forest", "Support Vector Machines", "AdaBoost Classifier"])

            if ml_model == "Support Vector Machines":
                
                # Create the support vector classifier model
                svm_model = svm.SVC()

                # Fit the model to the training data
                svm_model.fit(X1_train_scaled, final_y1_train)
                
                # Predict labels for testing features
                y1_prediction_svm = svm_model.predict(X1_test_scaled)
                
                # Generate classification report
                report = classification_report(final_y1_test, y1_prediction_svm, output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                # Display classification report in Streamlit
                st.write(report_df)
            
            elif ml_model == "Balanced Random Forest":
                
                # Instantiate a BalancedRandomForestClassifier instance
                brf = BalancedRandomForestClassifier(sampling_strategy='all', replacement=True)

                # Fit the model to the training data
                brf.fit(X1_train_scaled, final_y1_train)
                
                # Predict labels for testing features
                y1_prediction_brf = brf.predict(X1_test_scaled)
                
                # Generate classification report
                report = classification_report(final_y1_test, y1_prediction_brf, output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                # Display classification report in Streamlit
                st.write(report_df)
            
            elif ml_model == "AdaBoost Classifier":
                
                # Create the ada boost classifier
                adaboost = AdaBoostClassifier(n_estimators=50, random_state=0)
                
                # Fit the model to the training data
                adaboost.fit(X1_train_scaled, final_y1_train)
                
                # Predict labels for testing features
                y1_prediction_ada = adaboost.predict(X1_test_scaled)
                
                # Generate classification report
                report = classification_report(final_y1_test, y1_prediction_ada, output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                # Display classification report in Streamlit
                st.write(report_df)
            
        
        
        elif trading_algorithm == "RSI Oscillator":
            st.subheader("Relative Strength Index (RSI) Oscillator")
            
            st.write("The relative strength index (RSI) is a momentum indicator used in technical analysis. RSI measures the speed and magnitude of a security's recent price changes to evaluate overvalued or undervalued conditions in the price of that security.")

            st.write("The RSI can do more than point to overbought and oversold securities. It can also indicate securities that may be primed for a trend reversal or corrective pullback in price. It can signal when to buy and sell. Traditionally, an RSI reading of `70` or above indicates an overbought situation. A reading of `30` or below indicates an oversold condition.")
                     
            st.write("The trading signals are initialised to a neutral position and then start tracking trade signals and buy prices. The RSI thresholds (`rsi_lower_bound` and `rsi_upper_bound`) guide the buy and sell decisions: when the RSI falls below the lower threshold and no buy signal is active, it triggers a buy signal and records the buy price. Conversely, if the RSI rises above the upper threshold after a buy signal, it triggers a sell signal.") 

            st.write("Additionally, a stop-loss mechanism is in place: if the stock price falls below a stop-loss percentage from the buy price, it triggers a sell signal to limit potential losses. The algorithm ensures only one entry and exit point per trade cycle to effectively manage trade signals based on RSI values.")         
            
            # Fetch historical price data for the specified stock using yfinance
            stock_rsi_df = yf.download(selected_stock, start=start_date, end=end_date)

            # Extract the closing prices for each stock
            stock_rsi_df = pd.DataFrame(stock_rsi_df)
            
            # Option to select window size
            rsi_window_size = st.number_input("Enter the RSI Window Size", min_value=0, max_value=100, step=1, value=10)
            
            # Option to select upper bound
            rsi_upper_bound = st.number_input("Enter the RSI Upper Window", min_value=0, max_value=100, step=1, value=70)
            
            # Option to select lower bound
            rsi_lower_bound = st.number_input("Enter the RSI Lower Window", min_value=0, max_value=100, step=1, value=30)

            # Determine the Relative Strength Indicator (RSI) for the Dataset
            # Generate a series with the Relative Strength Indicator
            rsi_series = TA.RSI(stock_rsi_df, rsi_window_size)

            # Since TA uses the window size as part of the output column name for the RSI result, (eg "14 period RSI")
            # create a variable to make it easier to reference when generating our signals later on.

            # Assign the variable with the RSI column name
            rsi_column_name = f"RSI_{rsi_window_size}"

            # Rename the series
            rsi_series.name = rsi_column_name

            # Concatenate the RSI series to our stock dataset
            stock_rsi_df = pd.concat([stock_rsi_df, rsi_series], axis=1).dropna()

            # Option to select short and long window
            stop_percentage = st.number_input("Enter Stop Loss Percentage", min_value=0, max_value=100, step=1, value=10)

            # Create trading signals using the RSI algorithm

            # Initialise the Signal column to a neutral no action position
            stock_rsi_df["Signal"] = 0.0

            # Initialise the working trade signal and buy price variables
            trade_signal = 0
            buy_price = 0.0

            # Set the stop loss percentage (e.g., 10%  = 0.10)
            rsi_stop_loss_percentage = stop_percentage/100

            # Update the DataFrame Signal column 1 (entry) or -1 (exit) for a long position trading algorithm
            # where 1 is when the Close price is less than or equal to our RSI_LOWER_BOUND threshold
            # where -1 is when the Close price is greater or equal to our RSI_UPPER_BOUND threshold
            # Incorporate a conditional in the if-statement, to evaluate the value of the trade_signal so the algorithm 
            # plots only 1 entry and exit point per cycle.
            for index, row in stock_rsi_df[rsi_window_size:].iterrows():
                # Entry condition
                if (row[rsi_column_name] <= rsi_lower_bound) and (trade_signal < 1):
                    stock_rsi_df.loc[index, "Signal"] = 1.0
                    trade_signal = 1
                    buy_price = row["Close"]  # Record the buy price

                # Exit condition
                elif (row[rsi_column_name] >= rsi_upper_bound) and (trade_signal > 0):  
                    stock_rsi_df.loc[index, "Signal"] = 0
                    trade_signal = 0
                    buy_price = 0.0  # Reset the buy price

                    # Stop loss condition
                elif (buy_price > 0) and (row["Close"] < buy_price * (1 - rsi_stop_loss_percentage)) and (trade_signal > 0):
                    stock_rsi_df.loc[index, "Signal"] = 0
                    trade_signal = 0
                    buy_price = 0.0  # Reset the buy price


                # Maintain position
                else:
                    stock_rsi_df.loc[index, "Signal"] = trade_signal  # Maintain the position

            # Calculate the points in time when the Signal value changes
            # Identify trade entry (1) and exit (-1) points
            stock_rsi_df["Entry/Exit"] = stock_rsi_df["Signal"].diff()

            # Create traces
            trace_close = go.Scatter(x=stock_rsi_df.index, y=stock_rsi_df['Close'], mode='lines', name='Close')
            oscillator_trace = go.Scatter(x=stock_rsi_df.index, y=stock_rsi_df[rsi_column_name], mode='lines', name='Close')
            oversold_trace = go.Scatter(x=stock_rsi_df.index, y=[rsi_lower_bound] * len(stock_rsi_df), mode='lines', name='Oversold', line=dict(color='green', dash='dash'))
            overbought_trace = go.Scatter(x=stock_rsi_df.index, y=[rsi_upper_bound] * len(stock_rsi_df), mode='lines', name='Overbought', line=dict(color='red', dash='dash'))
            trace_entry = go.Scatter(x=stock_rsi_df[stock_rsi_df["Entry/Exit"] == 1.0].index, y=stock_rsi_df[stock_rsi_df["Entry/Exit"] == 1.0]['Close'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy')
            trace_exit = go.Scatter(x=stock_rsi_df[stock_rsi_df["Entry/Exit"] == -1.0].index, y=stock_rsi_df[stock_rsi_df["Entry/Exit"] == -1.0]['Close'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell')

            # Create figures
            fig = go.Figure()
            fig2 = go.Figure()

            # Add traces
            fig.add_trace(trace_close)
            fig.add_trace(trace_entry)
            fig.add_trace(trace_exit)

            fig2.add_trace(oscillator_trace)
            fig2.add_trace(oversold_trace)
            fig2.add_trace(overbought_trace)

            # Show plot
            st.plotly_chart(fig2)

            # Title
            st.subheader(f"{selected_stock} - Closing Price, Entry and Exit Points")

            # Show plot
            st.plotly_chart(fig)
            
            # Backtesting
            st.subheader("Backtesting")

            # Option to select initial
            initial_capital = st.number_input("Enter Initial Capital", min_value=1, step=1, value=10000)
            share_size = st.number_input("Enter Share Size", min_value=1, step=1, value=100)

            # Buy the chosen share position when the dual moving average crossover Signal equals 1
            # Otherwise, `Position` should be zero (sell)
            stock_rsi_df['Position'] = share_size * stock_rsi_df['Signal']

            # Determine the points in time where a 500 share position is bought or sold
            stock_rsi_df['Entry/Exit Position'] = stock_rsi_df['Position'].diff()

            # Multiply the close price by the number of shares held, or the Position
            stock_rsi_df['Portfolio Holdings'] = stock_rsi_df['Close'] * stock_rsi_df['Position']

            # Subtract the amount of either the cost or proceeds of the trade from the initial capital invested
            stock_rsi_df['Portfolio Cash'] = initial_capital - (stock_rsi_df['Close'] * stock_rsi_df['Entry/Exit Position']).cumsum() 

            # Calculate the total portfolio value by adding the portfolio cash to the portfolio holdings (or investments)
            stock_rsi_df['Portfolio Total'] = stock_rsi_df['Portfolio Cash'] + stock_rsi_df['Portfolio Holdings']

            # Calculate the portfolio daily returns
            stock_rsi_df['Portfolio Daily Returns'] = stock_rsi_df['Portfolio Total'].pct_change()

            # Calculate the portfolio cumulative returns
            stock_rsi_df['Portfolio Cumulative Returns'] = (1 + stock_rsi_df['Portfolio Daily Returns']).cumprod() - 1

            # Create traces
            total_portfolio_trace = go.Scatter(x=stock_rsi_df.index, y=stock_rsi_df['Portfolio Total'], mode='lines', name='Total Portfolio Value')

            exit_position_trace = go.Scatter(x=stock_rsi_df[stock_rsi_df['Entry/Exit'] == -1.0].index, y=stock_rsi_df[stock_rsi_df['Entry/Exit'] == -1.0]['Portfolio Total'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell')

            entry_position_trace = go.Scatter(x=stock_rsi_df[stock_rsi_df['Entry/Exit'] == 1.0].index, y=stock_rsi_df[stock_rsi_df['Entry/Exit'] == 1.0]['Portfolio Total'], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy')

            # Create figure
            fig3 = go.Figure()

            # Add traces
            fig3.add_trace(total_portfolio_trace)
            fig3.add_trace(exit_position_trace)
            fig3.add_trace(entry_position_trace)

            # Show plot
            st.plotly_chart(fig3)


            # Risk Reward Characteristics
            st.subheader("Risk-Reward Characteristics")

            # Calculate Annualized Returns
            annualized_returns = stock_rsi_df["Portfolio Daily Returns"].mean() * 252
            st.write(f"**Annualized Returns**: {annualized_returns}")

            # Calculate Cumulative Returns
            cumulative_returns = stock_rsi_df["Portfolio Cumulative Returns"][-1]
            st.write(f"**Cumulative Returns**: {cumulative_returns}")

            # Calculate Annual Volatility
            annual_volatility = stock_rsi_df["Portfolio Daily Returns"].std() * np.sqrt(252)
            st.write(f"**Annual Volatility**: {annual_volatility}")

            # Calculate Sharpe ratio
            sharpe_ratio = (stock_rsi_df["Portfolio Daily Returns"].mean() * 252) / (stock_rsi_df["Portfolio Daily Returns"].std() * np.sqrt(252))
            st.write(f"**Sharpe Ratio**: {sharpe_ratio}")


            # Machine Learning Implementation
            st.subheader("Machine Learning Model")

            st.write("The data is split into training and testing components using date offsets to create rolling windows (75-25 split). It is then standardised using `StandardScalar` function.")

            # Use the pct_change function to generate daily returns from the closing prices
            stock_rsi_df["Daily Returns"] = stock_rsi_df['Close'].pct_change()

            # Calculate the strategy returns and add them to the stock DataFrame
            stock_rsi_df['Strategy Returns'] = stock_rsi_df['Daily Returns'] * stock_rsi_df['Signal'].shift()

            # Assign a copy of the ema_fast and ema_slow columns to a features DataFrame called X1
            X1 = stock_rsi_df[[rsi_column_name]].shift().dropna()

            # Create the target set selecting the Signal column and assiging it to y1
            y1 = stock_rsi_df['Signal']

            # Date Offset to create train-test split
            train_duration = 3  # months
            step_size = 1  # month

            # Initialize lists to store training and testing dataframes
            all_X1_train = []
            all_y1_train = []
            all_X1_test = []
            all_y1_test = []

            # Initialize the starting point for the training period
            current_start = X1.index.min()

            # Loop until the end of the dataset
            while current_start + pd.DateOffset(months=train_duration) <= X1.index.max():
                # Define the training end period
                training_end = current_start + pd.DateOffset(months=train_duration)

                # Generate the training data for the current window
                X1_train = X1.loc[current_start:training_end]
                y1_train = y1.loc[current_start:training_end]
                all_X1_train.append(X1_train)
                all_y1_train.append(y1_train)

                # Define the testing period as the next time step after the training end
                testing_end = training_end + pd.DateOffset(months=step_size)
                if testing_end > X1.index.max():
                    testing_end = X1.index.max()

                # Generate the testing data for the current window
                X1_test = X1.loc[training_end:testing_end]
                y1_test = y1.loc[training_end:testing_end]
                all_X1_test.append(X1_test)
                all_y1_test.append(y1_test)

                # Update the start for the next iteration
                current_start = testing_end

            # Concatenate all training and testing dataframes
            final_X1_train = pd.concat(all_X1_train)
            final_y1_train = pd.concat(all_y1_train)
            final_X1_test = pd.concat(all_X1_test)
            final_y1_test = pd.concat(all_y1_test)

            # Scale the features DataFrames

            # Create a StandardScaler instance
            scaler1 = StandardScaler()

            # Apply the scaler model to fit the X1_train data
            X1_scaler = scaler1.fit(final_X1_train)

            # Transform the X_train and X_test DataFrames using the X1_scaler
            X1_train_scaled = X1_scaler.transform(final_X1_train)
            X1_test_scaled = X1_scaler.transform(final_X1_test)
            
            # Option to choose machine learning model
            ml_model = st.selectbox("Choose a Machine Learning Model", options=["Balanced Random Forest", "Support Vector Machines", "AdaBoost Classifier"])

            if ml_model == "Support Vector Machines":
                
                # Create the support vector classifier model
                svm_model = svm.SVC()

                # Fit the model to the training data
                svm_model.fit(X1_train_scaled, final_y1_train)
                
                # Predict labels for testing features
                y1_prediction_svm = svm_model.predict(X1_test_scaled)
                
                # Generate classification report
                report = classification_report(final_y1_test, y1_prediction_svm, output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                # Display classification report in Streamlit
                st.write(report_df)
            
            elif ml_model == "Balanced Random Forest":
                
                # Instantiate a BalancedRandomForestClassifier instance
                brf = BalancedRandomForestClassifier(sampling_strategy='all', replacement=True)

                # Fit the model to the training data
                brf.fit(X1_train_scaled, final_y1_train)
                
                # Predict labels for testing features
                y1_prediction_brf = brf.predict(X1_test_scaled)
                
                # Generate classification report
                report = classification_report(final_y1_test, y1_prediction_brf, output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                # Display classification report in Streamlit
                st.write(report_df)
            
            elif ml_model == "AdaBoost Classifier":
                
                # Create the ada boost classifier
                adaboost = AdaBoostClassifier(n_estimators=50, random_state=0)
                
                # Fit the model to the training data
                adaboost.fit(X1_train_scaled, final_y1_train)
                
                # Predict labels for testing features
                y1_prediction_ada = adaboost.predict(X1_test_scaled)
                
                # Generate classification report
                report = classification_report(final_y1_test, y1_prediction_ada, output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                # Display classification report in Streamlit
                st.write(report_df)

    except Exception as e:
        st.write(f"Something isn't Right!: {e}")

if __name__ == "__main__":
    main()
