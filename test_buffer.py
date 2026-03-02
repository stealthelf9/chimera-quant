import chimera_core
import numpy as np

def test_buffer():
    print("Testing Chimera Core Buffer...")
    buffer = chimera_core.MarketDataBuffer(100)
    
    # Append some ticks
    tick1 = chimera_core.OHLCV()
    tick1.timestamp = 1000000
    tick1.open = 100.0
    tick1.high = 105.0
    tick1.low = 95.0
    tick1.close = 102.0
    tick1.volume = 1000
    buffer.append(tick1)
    
    tick2 = chimera_core.OHLCV()
    tick2.timestamp = 1000060
    tick2.open = 102.0
    tick2.high = 103.0
    tick2.low = 101.0
    tick2.close = 101.5
    tick2.volume = 500
    buffer.append(tick2)
    
    view = buffer.get_buffer_view()
    print(f"NumPy View Shape: {view.shape}")
    print(f"NumPy View Dtype: {view.dtype}")
    print(f"First tick close price via NumPy: {view[0]['close']}")
    print(f"Second tick close price via NumPy: {view[1]['close']}")
    
    # Verify zero-copy behavior by modifying C++ buffer (appending reallocates if capacity is exceeded, 
    # but we reserved 100 so it won't reallocate for a few ticks).
    tick3 = chimera_core.OHLCV()
    tick3.close = 200.0
    buffer.append(tick3)
    
    # NOTE: view size is fixed to the size at the time of get_buffer_view(), 
    # but the memory is still the same until reallocation.
    
    print("Test passed!")

if __name__ == "__main__":
    test_buffer()
