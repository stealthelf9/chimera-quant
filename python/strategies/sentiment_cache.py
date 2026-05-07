import os
import json
import pandas as pd
from python.strategies.news_fetcher import AlpacaNewsFetcher
from python.strategies.llm_sentiment import LLMSentimentAnalyzer
from datetime import timedelta

class SentimentCache:
    def __init__(self, cache_file="data/sentiment_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.fetcher = AlpacaNewsFetcher()
        self.analyzer = LLMSentimentAnalyzer()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def fetch_and_cache_range(self, symbols: list, start_date: str, end_date: str):
        # We process in small chunks of days to respect limits and avoid excessive loading
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        current_dt = start_dt
        while current_dt <= end_dt:
            next_dt = min(current_dt + timedelta(days=7), end_dt)

            s_str = current_dt.strftime('%Y-%m-%d')
            e_str = next_dt.strftime('%Y-%m-%d')

            print(f"Fetching news for {s_str} to {e_str}")
            news_items = self.fetcher.fetch_news(symbols, s_str, e_str, limit=50)

            for item in news_items:
                for symbol in item['symbols']:
                    if symbol not in symbols: continue
                    # Extract date only to reduce granularity requirements
                    date_key = item['created_at'].strftime('%Y-%m-%d')
                    cache_key = f"{symbol}_{date_key}"

                    if cache_key not in self.cache:
                        print(f"Analyzing sentiment for {symbol} on {date_key}: {item['headline']}")
                        res = self.analyzer.analyze_headline(item['headline'])

                        score = 0.0
                        if res.get("sentiment") == "POSITIVE":
                            score = res.get("confidence", 0.0)
                        elif res.get("sentiment") == "NEGATIVE":
                            score = -res.get("confidence", 0.0)

                        self.cache[cache_key] = score
                        self.save_cache()

            current_dt = next_dt + timedelta(days=1)


    def get_sentiment(self, symbol, timestamp_str):
        # timestamp_str is assumed to be YYYY-MM-DD
        key = f"{symbol}_{timestamp_str}"
        if key in self.cache:
            return self.cache[key]
        return 0.0 # Neutral if missing
