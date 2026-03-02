from abc import ABC, abstractmethod
import chimera_core
from typing import Dict, Any

class BaseStrategy(ABC):
    """
    Plug-and-play base class to orchestrate generic backtesting loops and live forward-passes.
    """
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.buffer = chimera_core.MarketDataBuffer(100000)
    
    def on_live_tick(self, tick: chimera_core.OHLCV):
        """
        Called when a new live tick arrives from Alpaca.
        Appends the tick to the buffer and evaluates the strategy.
        """
        self.buffer.append(tick)
        self.evaluate()

    def load_historical(self, filepath: str):
        """
        Loads historical DBN ZSTD data for backtesting/training into the C++ buffer.
        """
        self.buffer.load_dbn(filepath)

    @abstractmethod
    def evaluate(self):
        """
        Contains the core logic for the strategy. Must be implemented by subclasses.
        This gets called on every new tick or during a backtest loop.
        """
        pass
