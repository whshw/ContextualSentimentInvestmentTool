import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
import urllib3
from bs4 import BeautifulSoup
import csv
from selenium.webdriver.common.action_chains import ActionChains
import requests
from urllib import request
import sys
from selenium import webdriver

df = pd.read_csv("../Data/MarketWatch List Countries.csv")

country_names = []
count = 0
for index, row in df.iloc[1:2].iterrows():
    sys.path.insert(0, 'chromedriver_mac_arm64/chromedriver')
    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--incognito')
    chrome_options.add_argument('--blink-settings=imagesEnabled=false')
    chrome_options.page_load_strategy = 'eager'
    driver = webdriver.Chrome('chromedriver', options=chrome_options)
    country_name = row['Country name']
    url = row['url']
    count = count + 1
    print(count)
    print(country_name)

    if not pd.isna(url):
        driver.get(url)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        wait = WebDriverWait(driver, 20)
        time.sleep(40)
        # try:
        #     # click yes i agree
        #     WebDriverWait(driver, 20).until(
        #         EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//iframe[@title='SP Consent Message']")))
        #     WebDriverWait(driver, 20).until(
        #         EC.element_to_be_clickable((By.XPATH, "//button[@title='YES, I AGREE']"))).click()
        #     # close the window
        #     WebDriverWait(driver, 30).until(
        #         EC.element_to_be_clickable((By.CSS_SELECTOR, "#cx-scrim-wrapper > button"))).click()
        # except:
        #     pass

        for i in range(0, 600):
            # For search
            element = driver.find_element(By.CSS_SELECTOR,
                                          '#maincontent > div > div.container.container--search-results > '
                                          'div.region.region--primary > div.column.column--primary > '
                                          'div.element.element--tabs > mw-tabs > div.element__body.j-tabPanes > '
                                          'div.tab__pane.is-active.j-tabPane > div > div.group.group--buttons.cover > '
                                          'button.btn.btn--secondary.js--more-headlines-site-search')
            sm = driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
            element.click()
            time.sleep(1)

        html = driver.page_source
        driver.quit()
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for link in soup.find_all('a', href=re.compile('(story|articles)')):
            links.append(link.get('href'))
        links = list(dict.fromkeys(links))
        links = [link for link in links if link is not None]
        # print(links)

        locals()["df" + str(country_name)] = pd.DataFrame(columns=['Date', 'Headline', 'Text'])
        for link in links:
            try:
                url = link
                html = request.urlopen(url).read().decode('utf-8')
                soup = BeautifulSoup(html, 'html.parser')

                head = soup.find('div', attrs={'class': 'article__masthead'})
                headline = head.find('h1').string

                if headline == None:
                    head = soup.find('div', attrs={'class': 'article__masthead'})
                    headline = head.find('h1', attrs={'class': 'article__headline'})
                    headline = headline.contents[2]

                date = head.find('time', attrs={'class': 'timestamp timestamp--pub'}).string
                text = soup.find('div', attrs={'id': 'js-article__body'})

                if text == None:
                    continue

                paragraphs = text.find_all('p')

                text_final = ""
                for paragraph in paragraphs:
                    text_final = text_final + paragraph.getText()

                # company_name=str(company_name).strip()
                date = " ".join(date.split())
                headline = " ".join(headline.split())
                text_final = " ".join(text_final.split())
                locals()["df2" + str(country_name)] = pd.DataFrame(
                    {"Date": [date], "Headline": [headline], "Text": [text_final]})
                locals()["df" + str(country_name)] = locals()["df" + str(country_name)].append(
                    locals()["df2" + str(country_name)])

            except:
                pass

        country_names.append(country_name)
        file_path = '../Data/Countries/{0}_articles.csv'.format(country_name)
        locals()["df" + str(country_name)].to_csv(file_path)
        locals()["df" + str(country_name)].to_csv('{0}.csv'.format(country_name))
