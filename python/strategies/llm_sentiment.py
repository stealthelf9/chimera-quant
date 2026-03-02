import requests
import json
from typing import Dict, Any

class LLMSentimentAnalyzer:
    """
    Interface for localized sentiment analysis via Ollama.
    Expects Ollama to be running locally on port 11434 with a lightweight model like llama3 or mistral.
    """
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        print(f"Initializing Local LLM Analyzer using {model_name} on {self.api_url}")

    def analyze_headline(self, headline: str) -> Dict[str, Any]:
        """
        Takes a news headline and requests the local LLM to return a JSON sentiment score.
        """
        prompt = (
            f"Analyze the following financial headline: '{headline}'. "
            "Respond strictly in JSON format with three keys: "
            "'sentiment' (string: POSITIVE, NEGATIVE, or NEUTRAL), "
            "'confidence' (float between 0.0 and 1.0), and "
            "'reasoning' (a very brief 1 sentence explanation)."
        )
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            # The Ollama API returns the generated string in the 'response' key
            # Since we requested format=json, the string should be parseable JSON
            return json.loads(result.get("response", "{}"))
        except Exception as e:
            print(f"Error querying local LLM: {e}")
            return {"sentiment": "NEUTRAL", "confidence": 0.0, "reasoning": str(e)}

if __name__ == "__main__":
    analyzer = LLMSentimentAnalyzer()
    
    # Example usage:
    test_headline = "Federal Reserve announces unexpected 50 basis point interest rate cut, markets rally"
    print(f"Testing headline: '{test_headline}'")
    
    # Note: This will fail if Ollama is not installed and running locally with the requested model.
    # We catch the exception gracefully so as to not break the trading loop.
    result = analyzer.analyze_headline(test_headline)
    print(f"Result: {result}")
