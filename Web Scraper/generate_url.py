import os
import csv

# Data
data = [
    ['Country name', 'url'],
    ['United States', 'https://www.marketwatch.com/search?q=United%20States&ts=0&tab=All%20News'],
    ['Canada', 'https://www.marketwatch.com/search?q=Canada&ts=0&tab=All%20News'],
    ['United Kingdom', 'https://www.marketwatch.com/search?q=United%20Kingdom&ts=0&tab=All%20News'],
    ['Australia', 'https://www.marketwatch.com/search?q=Australia&ts=0&tab=All%20News'],
    ['China', 'https://www.marketwatch.com/search?q=China&ts=0&tab=All%20News'],
    ['Denmark', 'https://www.marketwatch.com/search?q=Denmark&ts=0&tab=All%20News'],
    ['Finland', 'https://www.marketwatch.com/search?q=Finland&ts=0&tab=All%20News'],
    ['France', 'https://www.marketwatch.com/search?q=France&ts=0&tab=All%20News'],
    ['Germany', 'https://www.marketwatch.com/search?q=Germany&ts=0&tab=All%20News'],
    ['Japan', 'https://www.marketwatch.com/search?q=Japan&ts=0&tab=All%20News'],
    ['Italy', 'https://www.marketwatch.com/search?q=Italy&ts=0&tab=All%20News'],
    ['Netherlands', 'https://www.marketwatch.com/search?q=Netherlands&ts=0&tab=All%20News'],
    ['Norway', 'https://www.marketwatch.com/search?q=Norway&ts=0&tab=All%20News'],
    ['Portugal', 'https://www.marketwatch.com/search?q=Portugal&ts=0&tab=All%20News'],
    ['Singapore', 'https://www.marketwatch.com/search?q=Singapore&ts=0&tab=All%20News'],
    ['South Korea', 'https://www.marketwatch.com/search?q=South%20Korea&ts=0&tab=All%20News'],
    ['Spain', 'https://www.marketwatch.com/search?q=Spain&ts=0&tab=All%20News'],
    ['Sweden', 'https://www.marketwatch.com/search?q=Sweden&ts=0&tab=All%20News'],
    ['Switzerland', 'https://www.marketwatch.com/search?q=Switzerland&ts=0&tab=All%20News'],
    ['New Zealand', 'https://www.marketwatch.com/search?q=New%20Zealand&ts=0&tab=All%20News']
]

# Define file path
root_path = os.getcwd()
path = os.path.join(root_path, "Data/MarketWatch List Countries.csv")

# Create file and write data
with open(path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
