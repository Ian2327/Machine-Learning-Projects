import yfinance as yf
import matplotlib.pyplot as plt

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
sp500.index
#sp500.plot.line(y="Close", use_index=True)
del sp500["Dividends"]
del sp500["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"] #Don't include data that is from the future (i.e. Target, Tomorrow)
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])

import pandas as pd
preds = pd.Series(preds, index=test.index) #Convert numpy array to pandas series

score = precision_score(test["Target"], preds) #Compares past actual and predicted data

combined = pd.concat([test["Target"], preds], axis=1)
#combined.plot()

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

#Predictions beginning with the first 10 years (2500 days), then add 1 year
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)
predictions["Predictions"].value_counts() #If printed shows number of 0 and 1 in total

score = precision_score(predictions["Target"], predictions["Predictions"])
print("Precision (before): {}".format(score))

#percentage of days where the market increases
increase_percentage = predictions["Target"].value_counts() / predictions.shape[0]
print("Actual percent of days increase/decrease:\n{}".format(increase_percentage))

horizons = [2,5,60,250,1000] #2 days, 1 week, 3 months, 1 year, 4 years
new_predictors = []
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"] #ratio of today's close vs past previous days
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]
sp500 = sp500.dropna() #Drops all rows with NaN (begins in 1993)
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1] #Predicts and returns probability (0, 1) rather than 0 or 1
    preds[preds >= .6] = 1 #Only set to 1 if probability is 60% or greater
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined
predictions = backtest(sp500, model, new_predictors)
score = precision_score(predictions["Target"], predictions["Predictions"])
print("Predictions (with improved model): {}".format(score))
plt.show()