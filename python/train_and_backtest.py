import os
import sys
import zipfile

# Ensure CMake build output is in PYTHONPATH for the chimera_core C++ module
BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
sys.path.append(BUILD_DIR)

import chimera_core
from strategies.model import AIStrategy
from storage.cache import EvaluationCache

def main():
    print("=========================================")
    print(" Chimera Quant: End-to-End AI & Backtest ")
    print("=========================================")
    
    # 1. Look for Databento zip files in data/
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    zip_files = [f for f in os.listdir(data_dir) if f.endswith(".zip")]
    if not zip_files:
        print("No Databento ZIPs found in data/.")
        return

    # Extract a .dbn.zst from the first zip
    zip_path = os.path.join(data_dir, zip_files[0])
    print(f"Inspecting Archive: {zip_files[0]}")
    target_zst = None
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith(".dbn.zst"):
                target_zst = name
                print(f"Extracting {target_zst} to data/extracted/...")
                zf.extract(name, path=os.path.join(data_dir, "extracted"))
                break

    if not target_zst:
        print("No .dbn.zst files found in archive.")
        return

    dbn_file = os.path.join(data_dir, "extracted", target_zst)

    # 2. Load into C++ High-Performance Buffer
    print("\n--- Loading Data to Zero-Copy Buffer ---")
    data_engine = chimera_core.MarketDataBuffer(1000000)
    data_engine.load_dbn(dbn_file)
    total_ticks = len(data_engine.get_buffer_view())
    print(f"Loaded {total_ticks} ticks natively mapped to NumPy.")

    if total_ticks < 100:
        print("Not enough data to train. Exiting.")
        return

    # 3. Initialize PyTorch LSTM AI
    print("\n--- Initializing PyTorch AI ---")
    ai_strategy = AIStrategy(name="ChimeraNet_Alpha", params={"window_size": 30})
    
    # We split data 80% Train, 20% Backtest
    # This involves making sub-buffers. Since it's C++, we can slice the numpy view, 
    # but the AI needs to just look at a subset buffer.
    train_size = int(total_ticks * 0.8)
    
    # Create a training buffer natively
    ai_strategy.buffer = data_engine.slice(0, train_size)
    
    # 4. Train the Model Natively
    ai_strategy.train(epochs=2, batch_size=128)

    # 5. Execute Predictions over the 20% Validation Set
    print("\n--- Generating Predictions for Validation Set ---")
    validation_size = total_ticks - train_size
    signals = [0] * total_ticks # Pad the first 80% with 0s for generic alignment matching buffer sizes
    
    # To predict sequentially over the validation part naturally, we slice a moving window
    for i in range(train_size, total_ticks):
        ai_strategy.buffer = data_engine.slice(i - ai_strategy.window_size, i + 1)
        sig = ai_strategy.evaluate()
        if sig:
            signals[i] = sig

    # 6. Execute Native C++ Backtest via Python Pybind11 Interop
    print("\n--- Executing C++ Strategy Logic Simulator ---")
    stats = data_engine.run_backtest(
        initial_capital=100000.0,
        order_size=10000.0,
        size_is_percentage=False,
        commission=2.5,
        commission_is_percentage=False,
        slippage_penalty=0.005,
        start_timestamp=0, # Use whole range, but signals are 0 before train_size
        end_timestamp=0xFFFFFFFFFFFFFFFF, # Max
        signals=signals
    )
    
    print(f"\n[BACKTEST RESULTS]")
    print(f"Net Profit: ${stats.net_profit_usd:.2f} ({stats.net_profit_pct:.2f}%)")
    print(f"Win Rate: {stats.win_rate:.2f}%")
    print(f"Max Drawdown: {stats.max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {stats.sharpe_ratio:.2f}")

    # 7. Log to SQLite Cache
    print("\nCaching Results & Model Weights to SQLite...")
    cache = EvaluationCache()
    cache.log_backtest("ChimeraNet_Alpha", {"window_size": 30}, 0, 0, stats.net_profit_usd, stats.max_drawdown)
    cache.log_ai_weights("ChimeraNet_Alpha", 0.0, "weights/chimeranet_latest.pt", {"epochs": 2})
    
    # Optionally save state dictionaries here in real env
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights"), exist_ok=True)
    import torch
    torch.save(ai_strategy.model.state_dict(), os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "chimeranet_latest.pt"))

if __name__ == "__main__":
    main()
