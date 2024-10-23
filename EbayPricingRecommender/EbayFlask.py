from flask import Flask, request, render_template
from EbayScraper import get_ebay_listings
from EbayModel import load_model
import pandas as pd

app = Flask(__name__)

model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        product_name = request.form['product_name']

        # Fetch current listings from eBay
        listings = get_ebay_listings(product_name)

        prepared_data = listings[['condition']]
        predicted_prices = model.predict(prepared_data)
        listings['predicted_price'] = predicted_prices

        return render_template('results.html', listings=listings)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)