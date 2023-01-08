from bs4 import BeautifulSoup
import requests
import time
from tqdm import tqdm
import pandas as pd
import datetime
request_headers = {
'User-Agent' : ('Mozilla/5.0 (Windows NT 10.0;Win64; x64)\
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98\
Safari/537.36'), }
category = {100:'정치',101:'경제',102:'사회',103:'생활/문화',104:'세계',105:'IT/과학'}
# category = {100:'정치'}
def crawling_bot():
    crawled_news = {'headline':[],'summary':[],'content':[],'category':[],'datetime':[]}
    now = datetime.datetime.now()
    date = now.strftime('%Y-%m-%d')
    for hd_cate in category.keys():
        url = f'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1={hd_cate}'
        res = requests.get(url,headers = request_headers)
        soup = BeautifulSoup(
            res.text,
            'html.parser'
        )
        for td in tqdm(soup.find('div',attrs={"class": "list_body section_index"}).find_all('a')):
            try:
                news_url = td['href']
                news_res = requests.get(news_url,headers = request_headers)
                news_soup = BeautifulSoup(news_res.text,'html.parser')
                head = news_soup.find('div',attrs={'class':'media_end_head_title'}).text.strip()
                body = news_soup.find('div',attrs={'class':'go_trans _article_content'}).text.strip()
                summary = body.split('\n')[0]
                content = body.split('\n')[-1]
                crawled_news['headline'].append(head)
                crawled_news['summary'].append(summary)
                crawled_news['content'].append(body)
                crawled_news['category'].append(category[hd_cate])
                crawled_news['datetime'].append(date)

                time.sleep(.5)
            except:
                pass
    return crawled_news
if __name__ == '__main__':
    dt = crawling_bot()
    ori_data = pd.read_csv('./news.csv', sep='\t')
    ori_data.drop(columns = ['Unnamed: 0'],inplace=True)
    data = pd.DataFrame(dt)
    data = pd.concat([ori_data,data])
    data.to_csv('./news.csv',sep = '\t')
