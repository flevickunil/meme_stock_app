import plotly.express as px
import streamlit as st
import yfinance as yf
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class MemeStockVizualiser:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def plot_ticker_mentions(self):
        """
        Plots a scatter plot of ticker mention frequency over time.

        Parameters:
        tickers (DataFrame): A DataFrame containing 'date', 'count', and 'ticker' columns.
        """
        color_discrete_sequence = px.colors.qualitative.Plotly
        fig = px.scatter(self.dataframe, x='date', y='count', color='ticker',
                         title='Ticker Mention Frequency Over Time', color_discrete_sequence=color_discrete_sequence)
        st.plotly_chart(fig)

    def plot_ticker_cumsum(self):
        """
        Plots a line plot of cumulative ticker mention frequency over time.

        Parameters:
        tickers (DataFrame): A DataFrame containing 'date', 'count', and 'ticker' columns.
        """
        # Convert the date column to datetime format
        self.dataframe['date'] = pd.to_datetime(self.dataframe['date'], format="%Y-%m-%d")

        # Sort the dataframe by date and ticker
        self.dataframe = self.dataframe.sort_values(by=['ticker', 'date'])

        # Calculate cumulative sum for each ticker
        self.dataframe['cumulative_count'] = self.dataframe.groupby('ticker')['count'].cumsum()

        # Plot the cumulative data
        color_discrete_sequence = px.colors.qualitative.Plotly
        fig = px.line(self.dataframe, x='date', y='cumulative_count', color='ticker',
                      title='Cumulative Ticker Mention Frequency Over Time', color_discrete_sequence=color_discrete_sequence)
        st.plotly_chart(fig)

    def get_stock_info(self):
        st.title('Stock Ticker Frequency and Volume')

        # User inputs for date and ticker
        selected_date = st.selectbox('Select Date:', self.dataframe['date'].unique())
        selected_ticker = st.selectbox('Select Ticker:', self.dataframe['ticker'].unique())

        # Button to fetch trading volume and additional details
        if st.button('Show Details'):
            # Fetch trading volume data
            stock = yf.Ticker(selected_ticker)
            selected_date_dt = pd.to_datetime(selected_date)
            hist = stock.history(start=selected_date_dt, end=selected_date_dt + pd.DateOffset(days=30))
            hist.reset_index(names='date', inplace=True)

            # Fetch additional stock information
            info = stock.info
            business_sum = info.get('longBusinessSummary', 'No data')
            employee_num = info.get('fullTimeEmployees', 'No data')
            market_cap = info.get('marketCap', 'No data')
            currency = info.get('currency', 'No data')
            beta = info.get('beta', 'No data')
            industry = info.get('industry', 'No data')
            sector = info.get('sector', 'No data')

            # Display trading volume and additional information
            st.write(f'**Business Summary:** {business_sum}')
            st.write(f'**Industry:** {industry}')
            st.write(f'**Sector:** {sector}')
            st.write(f'**Number of Employees:** {format(int(employee_num), ",")}')
            st.write(f'**Market Cap:** {format(int(market_cap), ",")} {currency}')
            st.write(f'**Beta:** {beta}')

            # Display the frequency chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=hist['date'], y=hist['Volume'], name='Volume'), secondary_y=False)
            fig.add_trace(go.Scatter(x=hist['date'], y=hist['Close'], mode='lines+markers', name='Price', line=dict(color='orange')), secondary_y=True)
            fig.update_layout(title=f'30 day Traded Volume of {selected_ticker}', xaxis_title='Date', yaxis_title='Volume', yaxis2=dict(title='Price', overlaying='y', side='right'), legend=dict(x=0, y=1, traceorder='normal'), barmode='group')
            st.plotly_chart(fig)

# This ensures the script runs as a Streamlit app
if __name__ == '__main__':
    # get ticker reddit dataframe
    ticker_df = pd.read_csv('ticker_frequencies.csv')

    app = MemeStockVizualiser(ticker_df)
    st.title('Meme Stock App - View WSB Ticker Mentions & Corresponding Stock Volume Data')
    app.plot_ticker_mentions()
    app.plot_ticker_cumsum()
    app.get_stock_info()