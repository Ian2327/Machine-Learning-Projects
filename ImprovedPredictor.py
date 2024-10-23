import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, precision_recall_curve, recall_score, f1_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

sp500 = yf.Ticker("^GSPC").history(period="max")
sp500.index
#sp500.plot.line(y="Close", use_index=True)
del sp500["Dividends"]
del sp500["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()

for lag in range(1, 6):  # Create lag features for the past 5 days
    sp500[f"Close_Lag_{lag}"] = sp500["Close"].shift(lag)
    sp500[f"Volume_Lag_{lag}"] = sp500["Volume"].shift(lag)


# Beginning of old predictor
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"] # Don't include data that is from the future (i.e. Target, Tomorrow)
model.fit(train[predictors], train["Target"])

preds = model.predict(test[predictors])

preds = pd.Series(preds, index=test.index) # Convert numpy array to pandas series

score = precision_score(test["Target"], preds) # Compares past actual and predicted data

combined = pd.concat([test["Target"], preds], axis=1)
#combined.plot()

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Predictions beginning with the first 10 years (2500 days), then add 1 year
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)
predictions["Predictions"].value_counts() # If printed shows number of 0 and 1 in total

score = precision_score(predictions["Target"], predictions["Predictions"])
print("Precision (before): {}".format(score))

# Percentage of days where the market increases
increase_percentage = predictions["Target"].value_counts() / predictions.shape[0]
print("Actual percent of days increase/decrease:\n{}".format(increase_percentage))


# Beginning of new predictor
horizons = [2,5,60,250,1000] #2 days, 1 week, 3 months, 1 year, 4 years
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"] #ratio of today's close vs past previous days
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    volatility_column = f"Volatility_{horizon}"
    sp500[volatility_column] = sp500["Close"].rolling(horizon).std()
    max_close_column = f"Max_Close_{horizon}"
    sp500[max_close_column] = sp500["Close"].rolling(horizon).max()
    min_close_column = f"Min_Close_{horizon}"
    sp500[min_close_column] = sp500["Close"].rolling(horizon).min()
    new_predictors += [ratio_column, trend_column, max_close_column, min_close_column, volatility_column]

def RSI(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

sp500["RSI"] = RSI(sp500["Close"])
new_predictors.append("RSI")
sp500["VWAP"] = (sp500["Close"] * sp500["Volume"]).cumsum() / sp500["Volume"].cumsum()
new_predictors.append("VWAP")

sp500 = sp500.dropna() # Drops all rows with NaN (begins in 1993)

# Outlier Detection (Clip extreme outliers)
q_low = sp500["Close"].quantile(0.01)
q_high = sp500["Close"].quantile(0.99)
sp500.loc[:, "Close"] = sp500["Close"].copy().clip(lower=q_low, upper=q_high)

param_grid = {
    'n_estimators': [100, 200, 300],
    'min_samples_split': [50, 100, 150],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=1), param_grid, cv=5, scoring='precision')
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

grid_search.fit(train[predictors], train["Target"])

# Get the best model
model = grid_search.best_estimator_

# Alternatively, use ensemble model with VotingClassifier
ensemble_model = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)),
    ('gb', GradientBoostingClassifier(n_estimators=200, random_state=1)),
    ('lr', LogisticRegression())
], voting='soft')

model = ensemble_model  # Use this model for prediction and backtesting

#model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1] # Predicts and returns probability (0, 1) rather than 0 or 1
    precision, recall, thresholds = precision_recall_curve(test["Target"], preds)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[np.argmax(f1_scores)]
    preds[preds >= best_threshold] = 1
    preds[preds < best_threshold] = 0

    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Cross-Validation with TimeSeriesSplit and Backtesting
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    tscv = TimeSeriesSplit(n_splits=5)

    for train_index, test_index in tscv.split(data):
        train, test = data.iloc[train_index], data.iloc[test_index]
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)

predictions = backtest(sp500, model, new_predictors)
# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Train-test split
train_scaled = scaler.fit_transform(train[predictors])
test_scaled = scaler.transform(test[predictors])

# Update Logistic Regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, solver='lbfgs')  # Adjusted for convergence
model.fit(train_scaled, train["Target"])

# Predict and evaluate
preds = model.predict(test_scaled)
precision = precision_score(test["Target"], preds)
recall = recall_score(test["Target"], preds)
f1 = f1_score(test["Target"], preds)
accuracy = accuracy_score(test["Target"], preds)

# Print Metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Accuracy: {accuracy}")

# Predict whether the market will go up or down tomorrow
def predict_tomorrow(data, predictors, model):
    # Use the most recent row of data to predict tomorrow
    latest_data = data.iloc[-1:]  # Get the last row
    preds_proba = model.predict_proba(latest_data[predictors])[:, 1]  # Get probability for "up"
    
    # Set threshold at 0.5 (or adjust based on what you optimized earlier)
    prediction = 1 if preds_proba >= 0.5 else 0
    
    # Print the result
    if prediction == 1:
        print("The market is more likely to go UP tomorrow.")
    else:
        print("The market is more likely to go DOWN tomorrow.")

# Call the function with the trained model and latest data
predict_tomorrow(sp500, new_predictors, model)

#plt.show()