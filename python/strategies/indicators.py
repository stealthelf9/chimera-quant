import talib
import numpy as np
import chimera_core

class Indicators:
    """
    Wrapper class integrating TA-Lib directly on top of the zero-copy C++ arrays.
    """
    @staticmethod
    def rsi(buffer: chimera_core.MarketDataBuffer, timeperiod: int = 14) -> np.ndarray:
        view = buffer.get_buffer_view()
        if len(view) < timeperiod + 1:
            return np.array([])
        # TA-Lib RSI
        return talib.RSI(view['close'], timeperiod=timeperiod)

    @staticmethod
    def macd(buffer: chimera_core.MarketDataBuffer, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
        view = buffer.get_buffer_view()
        if len(view) < slowperiod + signalperiod:
            return np.array([]), np.array([]), np.array([])
        # TA-Lib MACD
        return talib.MACD(view['close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

    @staticmethod
    def ema(buffer: chimera_core.MarketDataBuffer, timeperiod: int = 30) -> np.ndarray:
        view = buffer.get_buffer_view()
        if len(view) < timeperiod:
            return np.array([])
        return talib.EMA(view['close'], timeperiod=timeperiod)

    @staticmethod
    def atr(buffer: chimera_core.MarketDataBuffer, timeperiod: int = 14) -> np.ndarray:
        view = buffer.get_buffer_view()
        if len(view) < timeperiod + 1:
            return np.array([])
        return talib.ATR(view['high'], view['low'], view['close'], timeperiod=timeperiod)
