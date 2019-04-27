from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import selenium as se


options = se.webdriver.ChromeOptions()
options.add_argument('headless')

browser = se.webdriver.Chrome(chrome_options=options)

browser.get("https://unsplash.com/search/photos/face")
#This code will scroll down to the end
for i in range(10):
    try:
        # Action scroll down
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        browser.implicitly_wait(1)
        browser.get("https://unsplash.com/search/photos/face")
        #break
    except:
        pass

html = browser.execute_script("return document.body.innerHTML")
print(html)

# timeout = 3
# r < 400 (200 300 redirection response)

soup = BeautifulSoup(html,'lxml')

for img in soup.findAll('a'):
    if img.has_attr('download'):
        print(img['href'])




