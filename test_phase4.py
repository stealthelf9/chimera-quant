import chimera_core
import numpy as np

def test_phase4():
    print("==== Phase 4 Dynamic Resampling & Backtesting Test ====")
    buffer = chimera_core.MarketDataBuffer()

    # Create 1440 1-minute ticks (1 Day of simulation)
    # The timestamp is represented in Nanoseconds (1 minute = 60,000,000,000)
    for i in range(1440):
        t = chimera_core.OHLCV()
        t.timestamp = i * 60_000_000_000
        t.open = 100.0
        t.high = 100.0 + (i * 0.1)
        t.low = 100.0 - (i * 0.1)
        t.close = 100.0 + (i % 2) # Fluctuating close
        t.volume = 100
        buffer.append(t)

    print(f"Original 1-Minute Buffer Size: {len(buffer.get_buffer_view())}")
    
    # Resample to 15m (1440 / 15 = 96)
    resampled_15m = buffer.resample(15)
    print(f"Resampled 15-Minute Buffer Size: {len(resampled_15m.get_buffer_view())}")
    assert len(resampled_15m.get_buffer_view()) <= 97
    
    # Resample to 1D (1440 / 1440 = 1)
    resampled_1d = buffer.resample(1440)
    print(f"Resampled 1-Day Buffer Size: {len(resampled_1d.get_buffer_view())}")
    
    print("\n==== Testing C++ Backtest Module ====")
    # Creating generic buy/sell signals matching buffer size exactly. Buy evens, Sell odds.
    signals = []
    for i in range(1440):
        if i % 10 == 0:
            signals.append(1)
        elif i % 10 == 5:
            signals.append(-1)
        else:
            signals.append(0)

    stats = buffer.run_backtest(
        initial_capital=100000.0,
        order_size=10000.0,
        size_is_percentage=False,
        commission=2.5,
        commission_is_percentage=False,
        slippage_penalty=0.005,
        start_timestamp=0,
        end_timestamp=1440 * 60_000_000_000,
        signals=signals
    )
    
    print(f"Net Profit: ${stats.net_profit_usd:.2f} ({stats.net_profit_pct:.2f}%)")
    print(f"Win Rate: {stats.win_rate:.2f}%")
    print(f"Max Drawdown: {stats.max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {stats.sharpe_ratio:.2f}")

if __name__ == "__main__":
    test_phase4()
