import os
import sys

# Ensure CMake build output is in PYTHONPATH for the chimera_core C++ module
BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "build")
sys.path.append(BUILD_DIR)

from python.strategies.model import AIStrategy
from python.strategies.live_data import MarketDataEngine
from python.strategies.executor import TradeExecutor
from python.storage.cache import EvaluationCache
from python.strategies.llm_sentiment import LLMSentimentAnalyzer

def main():
    print("=========================================")
    print(" Chimera Quant: Hybrid C++/Python Engine ")
    print("=========================================")
    
    # 1. Initialize High-Performance Engine
    # Load thick historical data into the C++ zero-copy buffer
    data_engine = MarketDataEngine(historical_file="data/sample.dbn.zst")
    
    # 2. Initialize the AI Strategy using ROCm GPU
    # The brain connects to the C++ buffer
    ai_strategy = AIStrategy(name="Chimera_Net_v1", params={"window_size": 60})
    ai_strategy.buffer = data_engine.buffer
    
    # 3. Initialize Execution System & Caches
    API_KEY = "PKOO9VTOEYVG4EBIIXXA"
    SECRET_KEY = "OLqMI9JtYfHbruwyzFDSMs9cmAdug5kAwFZJjyrJ"
    executor = TradeExecutor(api_key=API_KEY, secret_key=SECRET_KEY, paper=True)
    cache = EvaluationCache()
    llm = LLMSentimentAnalyzer(model_name="llama3")

    # 4. Simulate Backtest First (In reality, this would iterate historical ticks)
    print("\n--- Running Initial Feature Extraction & Inference ---")
    ai_strategy.evaluate()
    
    # 5. Enter Live Execution Mode
    # For now, disable to prevent hanging the terminal, but the architecture is ready.
    # print("\n--- Entering Live Execution Mode (Alpaca Paper) ---")
    # data_engine.stream.subscribe_trades(ai_strategy.on_live_tick, *["AAPL", "NVDA", "TSLA"])
    # data_engine.start_livestream()
    print("\nInitialization and Pipeline Linkage Successful.")

if __name__ == "__main__":
    main()
