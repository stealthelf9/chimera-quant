from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
import pandas as pd
from typing import List, Union
import os

class AlpacaNewsFetcher:
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
             print("Warning: Alpaca API keys not found. News fetching will fail.")
             self.client = None
        else:
             self.client = NewsClient(self.api_key, self.secret_key)

    def fetch_news(self, symbols: Union[str, List[str]], start_date: str, end_date: str, limit: int = 50):
        if not self.client:
            return []

        if isinstance(symbols, list):
            symbols = ",".join(symbols)

        try:
            req = NewsRequest(
                symbols=symbols,
                start=pd.to_datetime(start_date).tz_localize('UTC'),
                end=pd.to_datetime(end_date).tz_localize('UTC'),
                limit=limit,
                sort="DESC"
            )
            response = self.client.get_news(req)

            # Format the output into a more usable list of dicts
            news_data = []
            if hasattr(response, 'data') and 'news' in response.data:
                for article in response.data['news']:
                    news_data.append({
                        "headline": article.headline,
                        "summary": article.summary,
                        "created_at": article.created_at,
                        "symbols": article.symbols
                    })
            return news_data

        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

if __name__ == "__main__":
    fetcher = AlpacaNewsFetcher()
    news = fetcher.fetch_news(symbols=["AAPL", "TSLA"], start_date="2024-01-01", end_date="2024-01-07", limit=5)
    for n in news:
        print(f"{n['created_at']} | {n['symbols']} | {n['headline']}")
