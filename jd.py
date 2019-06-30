from selenium import webdriver;
from selenium.webdriver.support import expected_conditions as EC;
from selenium.webdriver.support.ui import WebDriverWait;
from selenium.webdriver.common.by import By;
import time;
browser = None;
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_argument("C:\\Users\\Administrator\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Login Data")
chrome_options.add_argument('--ignore-certificate-errors')
browser = webdriver.Chrome(chrome_options=chrome_options)
def openWindow():
	browser.get('http://jd.com')
	# C:\Users\Administrator\AppData\Local\Google\Chrome\User Data\Default\Login Data

def login():
	element = WebDriverWait(browser, 10).until(
		EC.presence_of_element_located((By.CLASS_NAME, "link-login"))
	)
	element.click();
	browser.find_element_by_class_name('pdl').click();
	time.sleep(1)
	# browser.switch_to.window(browser.window_handles[0])
	browser.switch_to_frame('ptlogin_iframe')
	# switcher_plogin
	loginByqq_element = WebDriverWait(browser, 30).until(
		EC.presence_of_element_located((By.ID, "switcher_plogin"))
	)
	loginByqq_element.click();
	password = ''
	userName = ''
	browser.find_element_by_id('u').send_keys(userName)
	browser.find_element_by_id('p').send_keys(password)
	browser.find_element_by_id('login_button').click();

openWindow()
login()
time.sleep()