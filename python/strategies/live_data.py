import os
import asyncio
# pyre-ignore-all-errors[21]
from alpaca.data.live import StockDataStream
from alpaca.data.models.trades import Trade
from alpaca.data.enums import DataFeed
import chimera_core

# Alpaca Credentials provided in instructions
API_KEY = "PKOO9VTOEYVG4EBIIXXA"
SECRET_KEY = "OLqMI9JtYfHbruwyzFDSMs9cmAdug5kAwFZJjyrJ"

# Top 10 Mega-Caps for live trading constraint
MEGA_CAPS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "LLY", "V"]

class MarketDataEngine:
    def __init__(self, historical_file: str | None = None):
        print("Initializing Market Data Engine...")
        self.buffer = chimera_core.MarketDataBuffer(1000000)
        
        if historical_file and os.path.exists(historical_file):
            print(f"Loading historical data from {historical_file}...")
            self.buffer.load_dbn(historical_file)
            print(f"Loaded {len(self.buffer.get_buffer_view())} historical records.")

        # Initialize Alpaca Data Stream (IEX for Free Tier)
        self.stream = StockDataStream(API_KEY, SECRET_KEY, feed=DataFeed.IEX)

    async def _handle_trade(self, data: Trade):
        """
        Callback for live trades.
        Stitches the live tick to the historical C-array via PyBind11.
        """
        tick = chimera_core.OHLCV()
        # Alpaca timestamps are datetime objects, convert to nanoseconds
        tick.timestamp = int(data.timestamp.timestamp() * 1e9)
        
        # Use tick price across OHLC. Volume is ignored for AI weights per constraints,
        # but stored in the struct. We can set it to 0 or data.size.
        tick.open = float(data.price)
        tick.high = float(data.price)
        tick.low = float(data.price)
        tick.close = float(data.price)
        tick.volume = int(data.size)
        
        self.buffer.append(tick)
        
        # Example to show the buffer is growing and view is zero-copy
        view = self.buffer.get_buffer_view()
        print(f"[{data.symbol}] Tick stitched. New buffer size: {len(view)}. Price: {tick.close}")

    def start_livestream(self):
        print(f"Subscribing to live IEX trades for {MEGA_CAPS}...")
        self.stream.subscribe_trades(self._handle_trade, *MEGA_CAPS)
        
        # Run the stream (this is a blocking call in alpaca-py)
        print("Starting stream...")
        self.stream.run()

if __name__ == "__main__":
    engine = MarketDataEngine()
    engine.start_livestream()
