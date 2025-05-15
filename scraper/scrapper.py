import random
import time
import requests
import psycopg2
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

class NewsScraper:
    def __init__(self):
        self.id = 0
        self.seen_links = set()
        self.session = requests.Session()
        retry = requests.packages.urllib3.util.retry.Retry(total=3, backoff_factor=1)
        adapter = requests.adapters.HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.conn = psycopg2.connect(
            dbname="news_db",
            user="affan",
            password="pass123",
            host="postgres",
            port="5432"
        )
        self.cursor = self.conn.cursor()

    def _save_article(self, title, content, category, link):
        if link in self.seen_links:
            return
        try:
            source = link.split('/')[2].split('.')[0]
            self.cursor.execute("""
                INSERT INTO labeled_articles (title, content, gold_label, source, timestamp)
                VALUES (%s, %s, %s, %s, NOW());
            """, (title, content, category, source))
            self.conn.commit()
            self.seen_links.add(link)
            self.id += 1
        except Exception as e:
            print(f"[DB ERROR] {e}")


    # EXPRESS
    def get_express_articles(self, max_pages=1):
        base_url = 'https://www.express.pk'
        categories = ['saqafat', 'business', 'sports', 'science', 'world']
        for category in categories:
            for page in range(1, max_pages + 1):
                print(f"[EXPRESS] Page {page} - {category}")
                try:
                    url = f"{base_url}/{category}/archives?page={page}"
                    time.sleep(random.uniform(1.5, 3.5))
                    response = self.session.get(url)
                    soup = BeautifulSoup(response.text, "html.parser")
                    cards = soup.find('ul', class_='tedit-shortnews listing-page').find_all('li')
                    for card in cards:
                        try:
                            div = card.find('div', class_='horiz-news3-caption')
                            headline = div.find('a').get_text(strip=True)
                            link = div.find('a')['href']
                            if link in self.seen_links:
                                continue
                            article_response = self.session.get(link)
                            content_soup = BeautifulSoup(article_response.text, "html.parser")
                            paras = content_soup.find('span', class_='story-text').find_all('p')
                            content = " ".join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
                            clean_cat = category.replace('saqafat', 'entertainment').replace('science', 'science-technology')
                            self._save_article(headline, content, clean_cat, link)
                        except Exception as e:
                            print(f"[EXPRESS ERROR] {e}")
                except Exception as e:
                    print(f"[EXPRESS FAIL] {e}")

    # JANG
    def get_jang_articles(self, max_pages=1):
        base_url = 'https://jang.com.pk'
        categories = ['entertainment', 'sports', 'international', 'business', 'science']
        for category in categories:
            for page in range(1, max_pages + 1):
                print(f"[JANG] Page {page} - {category}")
                try:
                    url = f"{base_url}/category/{category}/page/{page}"
                    time.sleep(random.uniform(1.5, 3.5))
                    response = self.session.get(url)
                    soup = BeautifulSoup(response.text, "html.parser")
                    cards = soup.find_all('a', href=True)
                    for card in cards:
                        try:
                            title_tag = card.find('h2')
                            if not title_tag:
                                continue
                            link = card.get('href')
                            if not link or link in self.seen_links:
                                continue
                            title = title_tag.get_text(strip=True)
                            article_response = self.session.get(link)
                            soup2 = BeautifulSoup(article_response.text, 'html.parser')
                            content_div = soup2.find('div', class_='detail_view_content')
                            paras = content_div.find_all('p') if content_div else []
                            content = " ".join(p.get_text(strip=True) for p in paras)
                            self._save_article(title, content, category, link)
                        except Exception as e:
                            print(f"[JANG ERROR] {e}")
                except Exception as e:
                    print(f"[JANG FAIL] {e}")

    # DUNYA
    def get_dunya_articles(self, max_pages=1):
        base_url = 'https://urdu.dunyanews.tv'
        categories = ['entertainment', 'sports', 'international', 'business', 'technology']
        unwanted = ['health', 'fakenews', 'pakistan', 'weirdnews']
        for category in categories:
            for page in range(1, max_pages + 1):
                print(f"[DUNYA] Page {page} - {category}")
                try:
                    page_url = f"{base_url}/index.php/ur/{category.capitalize()}?page={page}"
                    time.sleep(random.uniform(1.5, 3.5))
                    response = self.session.get(page_url)
                    soup = BeautifulSoup(response.text, "html.parser")
                    articles = soup.find_all('h3')
                    for article_tag in articles:
                        try:
                            a_tag = article_tag.find('a')
                            if not a_tag or not a_tag.get('href'):
                                continue
                            relative_link = a_tag['href']
                            article_url = f"{base_url}{relative_link}"
                            if article_url in self.seen_links or any(uw in article_url for uw in unwanted):
                                continue
                            article_response = self.session.get(article_url)
                            soup2 = BeautifulSoup(article_response.text, 'html.parser')
                            title_tag = soup2.find('div', class_='newsBox news-content')
                            title = title_tag.find('h1').get_text(strip=True) if title_tag else "No title"
                            content_div = soup2.find('div', class_='newsBox news-content')
                            paras = content_div.find_all('p') if content_div else []
                            content = " ".join(p.get_text(strip=True) for p in paras)
                            self._save_article(title, content, category, article_url)
                        except Exception as e:
                            print(f"[DUNYA ERROR] {e}")
                except Exception as e:
                    print(f"[DUNYA FAIL] {e}")

    # SAMAA
    def get_samaa_articles(self, max_pages=1):
        base_url = 'https://urdu.samaa.tv'
        categories = {
            'entertainment': 'https://urdu.samaa.tv/lifestyle',
            'sports': 'https://urdu.samaa.tv/sports',
            'business': 'https://urdu.samaa.tv/money',
            'international': 'https://urdu.samaa.tv/global',
            'technology': 'https://urdu.samaa.tv/tech'
        }
        headers = {'User-Agent': 'Mozilla/5.0'}
        for category, url in categories.items():
            for page in range(1, max_pages + 1):
                print(f"[SAMAA] Page {page} - {category}")
                try:
                    page_url = f"{url}?page={page}"
                    time.sleep(random.uniform(1.5, 3.5))
                    response = self.session.get(page_url, headers=headers)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    articles = soup.find_all('article', class_='story-article')
                    for article in articles:
                        try:
                            h3 = article.find('h3')
                            a_tag = h3.find('a') if h3 else None
                            if not a_tag:
                                continue
                            relative_link = a_tag['href']
                            article_url = f"{base_url}{relative_link}" if relative_link.startswith('/') else relative_link
                            if article_url in self.seen_links:
                                continue
                            article_response = self.session.get(article_url, headers=headers)
                            soup2 = BeautifulSoup(article_response.text, 'html.parser')
                            title = a_tag.get_text(strip=True)
                            content_div = soup2.find('div', class_='article-content')
                            content = " ".join(p.get_text(strip=True) for p in content_div.find_all('p')) if content_div else ""
                            self._save_article(title, content, category, article_url)
                        except Exception as e:
                            print(f"[SAMAA ERROR] {e}")
                except Exception as e:
                    print(f"[SAMAA FAIL] {e}")

    # def scrape_all(self, max_pages=1):
    #     print("[STARTING LABELED SCRAPE]")
    #     self.get_express_articles(max_pages)
    #     self.get_jang_articles(max_pages)
    #     self.get_dunya_articles(max_pages)
    #     self.get_samaa_articles(max_pages)
    #     print("[SCRAPING COMPLETE]")

    #     print("[LOADING RAW DATA]")
    #     df = pd.read_sql("SELECT * FROM labeled_articles;", self.conn)

    #     print("[CLEANING DATA]")
    #     df['gold_label'] = df['gold_label'].str.lower().map(category_mapping)
    #     df.dropna(subset=['content', 'gold_label'], inplace=True)
    #     df['content'] = df['content'].apply(clean_text)
    #     df['title'] = df['title'].apply(clean_text)
    #     df['label'] = pd.Categorical(
    #         df['gold_label'],
    #         categories=['entertainment', 'business', 'sports', 'science-technology', 'international'],
    #         ordered=True
    #     ).codes + 1
    #     df.drop(columns=['gold_label'], inplace=True)

    #     print("[INSERTING CLEANED DATA]")
    #     self.cursor.execute("""
    #         CREATE TABLE IF NOT EXISTS cleaned_articles (
    #             title TEXT,
    #             content TEXT,
    #             label INT,
    #             source TEXT,
    #             timestamp TIMESTAMP
    #         );
    #     """)
    #     for _, row in df.iterrows():
    #         self.cursor.execute("""
    #             INSERT INTO cleaned_articles (title, content, label, source, timestamp)
    #             VALUES (%s, %s, %s, %s, %s);
    #         """, (row['title'], row['content'], int(row['label']), row['source'], row['timestamp']))
    #     self.conn.commit()

    #     self.cursor.close()
    #     self.conn.close()
    #     print("[PIPELINE COMPLETE]")

    def scrape_all(self, max_pages=1):
        print("[STARTING LABELED SCRAPE]")
        self.get_express_articles(max_pages)
        self.get_jang_articles(max_pages)
        self.get_dunya_articles(max_pages)
        self.get_samaa_articles(max_pages)
        print("[SCRAPING COMPLETE]")
        
        # Close DB connection after scraping
        self.cursor.close()
        self.conn.close()

if __name__ == "__main__":
    try:
        scraper = NewsScraper()
        scraper.scrape_all(max_pages=1)
        print("[SUCCESS] Scraper pipeline finished.")
    except Exception as e:
        print(f"[FATAL ERROR] Pipeline failed: {e}")

