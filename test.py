import numpy as np
import pandas as pd

file_path = "Data.csv"
data = pd.read_csv(file_path, sep=';')

# Now let's drop the 'Volume' and 'Adj Close' columns as requested
data = data.drop(columns=['Volume', 'Adj Close'])

# Convert the Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Sort data by date to ensure it's in the right order
data = data.sort_values('Date')

# Calculate daily returns based on the 'Close' prices
data['Return'] = data['Close'].pct_change()

data['Volatility'] = data['Return'].rolling(window=30).std() * (252 ** 0.5)

data['MA_10'] = data['Close'].rolling(window=10).mean()  # 10-day moving average
data['MA_50'] = data['Close'].rolling(window=50).mean()  # 50-day moving average

# Price momentum (rate of change)
data['Momentum_10'] = data['Close'].pct_change(periods=10)  # 10-day momentum
data['Momentum_50'] = data['Close'].pct_change(periods=50)  # 50-day momentum

# Shift the volatility to predict the next period's volatility
data['Next_Volatility'] = data['Volatility'].shift(-1)

# Drop any rows with missing values (NaNs) that were created due to rolling windows and shifts
data_cleaned = data.dropna()

# Show the prepared dataset with features and the target (Next_Volatility)
data_cleaned[['Date', 'Close', 'MA_10', 'MA_50', 'Momentum_10', 'Momentum_50', 'Volatility', 'Next_Volatility']].head()

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Define the features (X) and target (y)
X = data_cleaned[['MA_10', 'MA_50', 'Momentum_10', 'Momentum_50', 'Volatility']]
y = data_cleaned['Next_Volatility']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the evaluation results
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Optionally, display the feature importances
feature_importances = rf_model.feature_importances_
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance:.4f}")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Select the features and target for training
X = data_cleaned_no_nan[['MA_10', 'MA_50', 'Momentum_10', 'Momentum_50', 'Volatility']]
y = data_cleaned_no_nan['Volatility'].shift(-1).dropna()  # Shift target for next period's volatility

# Make sure to align X and y by removing NaN values created by the shift
X = X.iloc[:-1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

def simulate_gbm_with_dynamic_volatility(S0, r, T, M, I, model, data):
    """
    Simulates Monte Carlo trajectories using dynamically predicted volatility.
    
    Parameters:
    S0 : float : Initial price of the asset
    r : float : Risk-free rate
    T : float : Time to maturity
    M : int : Number of time steps
    I : int : Number of simulations
    model : RandomForestRegressor : Trained model for predicting volatility
    data : DataFrame : Historical data for feature calculations
    
    Returns:
    S : ndarray : Simulated price paths
    """
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0  # Start the simulation from the actual current price
    
    # Loop over each simulation
    for i in range(I):
        # For each simulation, start with the last known features in the dataset
        sim_data = data.iloc[-10:].copy()  # We need at least 10 rows to compute rolling metrics
        
        for t in range(1, M + 1):
            # Prepare the features for prediction (based on the most recent data)
            features = sim_data[['MA_10', 'MA_50', 'Momentum_10', 'Momentum_50', 'Volatility']].values[-1:]

            # Convert features to DataFrame with column names
            feature_columns = ['MA_10', 'MA_50', 'Momentum_10', 'Momentum_50', 'Volatility']
            features_df = pd.DataFrame(features, columns=feature_columns)

            # Predict the volatility for the current step
            predicted_volatility = model.predict(features_df)[0]
            
            # Generate a random number from a normal distribution
            Z = np.random.standard_normal()
            
            # Simulate the next price using the predicted volatility
            S[t, i] = S[t - 1, i] * np.exp((r - 0.5 * predicted_volatility**2) * dt + predicted_volatility * np.sqrt(dt) * Z)
            
            # Add the new simulated price (Close) to sim_data
            new_close = S[t, i]
            new_row = pd.DataFrame({
                'Date': [sim_data['Date'].max() + pd.Timedelta(days=1)],
                'Close': [new_close]
            })
            
            # Concatenate the new row to sim_data to update the data
            sim_data = pd.concat([sim_data, new_row], ignore_index=True)

            # Recalculate the features based on the updated sim_data (rolling calculations)
            sim_data['MA_10'] = sim_data['Close'].rolling(window=10).mean()
            sim_data['MA_50'] = sim_data['Close'].rolling(window=50).mean()
            sim_data['Momentum_10'] = sim_data['Close'].pct_change(periods=10)
            sim_data['Momentum_50'] = sim_data['Close'].pct_change(periods=50)
            sim_data['Volatility'] = sim_data['Close'].pct_change().rolling(window=30).std() * (252 ** 0.5)  # Annualized volatility
    
    return S

# Parameters for Monte Carlo
r = 0.0334      # Risk-free rate
T = 1.0       # Time to maturity (1 year)
M = 252       # Number of time steps (daily)
I = 5         # Number of simulations

# Get the most recent closing price from the dataset as the initial price
S0 = data_cleaned_no_nan['Close'].iloc[-1]  # This should be 5528 based on your dataset

# Assuming the RandomForestRegressor model is already trained
# rf_model = RandomForestRegressor(...)  # Train the model as described in previous steps

# Run the Monte Carlo simulation with dynamically predicted volatility
simulated_prices = simulate_gbm_with_dynamic_volatility(S0, r, T, M, I, rf_model, data_cleaned_no_nan)

K = 5550   # Strike price
r = 0.0334  # Risk-free rate
T = 1.0   # Time to maturity (1 year)

# Get the final prices at maturity (the last row in the simulated prices)
final_prices = simulated_prices[-1, :]

# Calculate the payoff for a call option (for each path)
payoff = np.maximum(final_prices - K, 0)

# Discount the average payoff to today
option_price = np.exp(-r * T) * np.mean(payoff)

print(f"The price of the European call option is: {option_price:.4f}")