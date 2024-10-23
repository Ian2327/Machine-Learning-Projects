import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_model(data):
    X = data[['condition', 'seller_rating', 'listing_time']]
    Y = data['price']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(Y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    joblib.dump(model, 'price_model.pkl')

def load_model():
    model = joblib.load('price_model.pkl')
    return model