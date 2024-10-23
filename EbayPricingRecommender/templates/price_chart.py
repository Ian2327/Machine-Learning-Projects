import matplotlib.pyplot as plt

def create_price_chart(data):
    plt.figure(figsize=(10,5))
    plt.plot(data['listing_time'], data['price'], marker='o')
    plt.title('Price Trend')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.savefig('price_chart.png')