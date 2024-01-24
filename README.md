# EXECUTE3.0HACKATHON
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Sample historical data (replace with actual data from APIs)
historical_data = {
    'Date': ['2021-01-01', '2021-01-02', '2021-01-03'],
    'BTC_Price': [30000, 31000, 32000],
    'ETH_Price': [1000, 1100, 1200],
    # Add more cryptocurrencies and relevant features
}

user_preferences = {
    'Risk_Appetite': 0.7,
    'Preferred_Cryptos': ['BTC', 'ETH'],
    # Add more user preferences
}

# Create a DataFrame from historical data
df = pd.DataFrame(historical_data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Feature engineering - Calculate daily returns
df['BTC_Return'] = df['BTC_Price'].pct_change()
df['ETH_Return'] = df['ETH_Price'].pct_change()

# Merge user preferences with historical data
df_user = pd.DataFrame(user_preferences, index=df.index)
df_merged = pd.concat([df, df_user], axis=1).dropna()

# Define features and target variable
features = ['BTC_Return', 'ETH_Return', 'Risk_Appetite']
target = 'BTC_Price'  # Predict BTC Price

# Split data into training and testing sets
train_size = int(0.8 * len(df_merged))
train, test = df_merged.iloc[:train_size], df_merged.iloc[train_size:]

# Scale features
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train[features])
test_scaled = scaler.transform(test[features])

# Train a linear regression model
model = LinearRegression()
model.fit(train_scaled, train[target])

# Make predictions on the test set
test['Predicted_BTC_Price'] = model.predict(test_scaled)

# Evaluate the model
mse = mean_squared_error(test['BTC_Price'], test['Predicted_BTC_Price'])
print(f'Mean Squared Error: {mse}')

# Sample personalized recommendation
user_input = scaler.transform([[0.02, 0.01, 0.8]])  # Replace with actual user input
predicted_price = model.predict(user_input)[0]
print(f'Predicted BTC Price based on user preferences: {predicted_price}')
