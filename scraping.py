from pygooglenews import GoogleNews
from pydantic import BaseModel, Field
from firecrawl import FirecrawlApp
import time
import os 
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

class StockNewsArticle(BaseModel):
    title: str = Field(description="The title of the news article")
    content: str = Field(description="The full content of the news article")

def fetch_stock_news_links(query, max_articles=3):
    """Fetches news links related to a stock ticker using PyGoogleNews."""
    gn = GoogleNews()
    search_results = gn.search(query)

    news_links = []
    count = 0

    for entry in search_results["entries"]:
        if count >= max_articles:
            break

        news_links.append({
            "title": entry.title,
            "url": entry.link,
            "published_at": entry.published
        })
        count += 1

    return {"query": query, "articles": news_links}

def extract_full_stock_news(articles):
    """Extracts full news content using FireCrawl for a given list of articles."""
    app = FirecrawlApp(api_key=firecrawl_api_key)  # Initialize FireCrawl
    extracted_news = []

    for article in articles:
        url = article["url"]
        try:
            data = app.scrape_url(
                url,
                params={
                    "formats": ["extract"],
                    "extract": {
                        "schema": StockNewsArticle.model_json_schema()
                    },
                    "actions": [
                        {"type": "wait", "milliseconds": 2000},
                        {"type": "scroll", "behavior": "smooth"}
                    ]
                }
            )

            extracted_data = data.get("extract", {})
            extracted_news.append({
                "title": extracted_data.get("title", article["title"]),
                "content": extracted_data.get("content", "Content not available.")
            })
        except Exception as e:
            print(f"Error extracting {url}: {e}")

        time.sleep(1.5)

    return {"articles": extracted_news}