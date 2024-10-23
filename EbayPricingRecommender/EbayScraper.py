import requests
from bs4 import BeautifulSoup
import pandas as pd
import argparse

def get_ebay_listings(product_name):
    url = f"https://www.ebay.com/sch/i.html?_nkw={product_name}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    listings = []
    for item in soup.find_all('li', class_='s-item s-item__pl-on-bottom'):
        try:
            title = item.find('div', class_='s-item__title').text
            price = item.find('span', class_='s-item__price').text
            condition = item.find('span', class_='SECONDARY_INFO').text
            listings.append({'title': title, 'price': price, 'condition': condition})
        except AttributeError:
            continue
        
    return pd.DataFrame(listings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("product_name", type=str)
    args = parser.parse_args()
    product_data = get_ebay_listings(args.product_name)
    product_data.to_csv(f'ebay_listings_{args.product_name}.csv', index=False)
    print(f"Scraped data saved to ebay_listings_{args.product_name}.csv")