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
predictions["Predictions"].value_counts()

score = precision_score(predictions["Target"], predictions["Predictions"])
print(score)
plt.show()