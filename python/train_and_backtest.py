import os
import sys
import zipfile
import argparse
from datetime import datetime

# Ensure CMake build output is in PYTHONPATH for the chimera_core C++ module
BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
sys.path.append(BUILD_DIR)

import chimera_core
from python.strategies.model import AIStrategy
from python.storage.cache import EvaluationCache

def parse_args():
    parser = argparse.ArgumentParser(description="Chimera Quant Backtest & Training CLI")
    parser.add_argument("--mode", choices=["train", "backtest", "live"], default="backtest", help="Execution mode")
    parser.add_argument("--strategies", type=str, default="ai", help="Comma separated list of strategies")
    parser.add_argument("--timeframe", type=str, default="1m", help="Resampling timeframe (e.g., 1m, 5m, 1h, 1d)")
    parser.add_argument("--start-date", type=str, default="", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="", help="End date YYYY-MM-DD")
    parser.add_argument("--symbols", type=str, default="ALL", help="Comma separated symbols or ALL")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--position-size", type=float, default=0.1, help="Position size (0.1 = 10 percent of portfolio)")
    parser.add_argument("--slippage", type=float, default=0.005, help="Slippage penalty")
    return parser.parse_args()

def date_to_nanos(date_str: str) -> int:
    if not date_str:
        return 0
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1e9)

def main():
    args = parse_args()
    print("=========================================")
    print(f" Chimera Quant: Mode -> {args.mode.upper()} ")
    print("=========================================")
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    zip_files = [f for f in os.listdir(data_dir) if f.endswith(".zip")]
    if not zip_files:
        print("No Databento ZIPs found in data/.")
        return

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

    print("\n--- Loading Data to Zero-Copy Buffer ---")
    data_engine = chimera_core.MarketDataBuffer(1000000)
    data_engine.load_dbn(dbn_file)
    total_ticks = len(data_engine.get_buffer_view())
    print(f"Loaded {total_ticks} ticks natively mapped to NumPy.")

    if total_ticks < 100:
        print("Not enough data to train/backtest. Exiting.")
        return

    # Process Resampling
    interval_minutes = 1
    if args.timeframe.endswith("m"):
        interval_minutes = int(args.timeframe[:-1])
    elif args.timeframe.endswith("h"):
        interval_minutes = int(args.timeframe[:-1]) * 60
    elif args.timeframe.endswith("d"):
        interval_minutes = int(args.timeframe[:-1]) * 1440
        
    if interval_minutes > 1:
        data_engine = data_engine.resample(interval_minutes)
        total_ticks = len(data_engine.get_buffer_view())
        print(f"Resampled native buffer to {args.timeframe}. New tick count: {total_ticks}")

    print("\n--- Initializing PyTorch AI ---")
    ai_strategy = AIStrategy(name="ChimeraNet_Alpha", params={"window_size": 30})
    
    train_size = int(total_ticks * 0.8)
    ai_strategy.buffer = data_engine.slice(0, train_size)
    
    if args.mode in ["train", "backtest"]:
        ai_strategy.train(epochs=2, batch_size=128)

    print("\n--- Generating Predictions for Validation Set ---")
    validation_size = total_ticks - train_size
    signals = [0] * total_ticks
    
    for i in range(train_size, total_ticks):
        ai_strategy.buffer = data_engine.slice(i - ai_strategy.window_size, i + 1)
        sig = ai_strategy.evaluate()
        if sig:
            signals[i] = sig

    print("\n--- Executing C++ Strategy Logic Simulator ---")
    start_ns = date_to_nanos(args.start_date)
    end_ns = date_to_nanos(args.end_date) if args.end_date else 0xFFFFFFFFFFFFFFFF
    
    stats = data_engine.run_backtest(
        initial_capital=args.capital,
        order_size=args.position_size,
        size_is_percentage=True,
        commission=2.5,
        commission_is_percentage=False,
        slippage_penalty=args.slippage,
        start_timestamp=start_ns,
        end_timestamp=end_ns,
        signals=signals
    )
    
    print(f"\n[BACKTEST RESULTS]")
    print(f"Net Profit: ${stats.net_profit_usd:.2f} ({stats.net_profit_pct:.2f}%)")
    print(f"Win Rate: {stats.win_rate:.2f}%")
    print(f"Max Drawdown: {stats.max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {stats.sharpe_ratio:.2f}")

    print("\nCaching Results & Model Weights to SQLite...")
    cache = EvaluationCache()
    cache.log_backtest("ChimeraNet_Alpha", {"window_size": 30}, 0, 0, stats.net_profit_usd, stats.max_drawdown)
    cache.log_ai_weights("ChimeraNet_Alpha", 0.0, "weights/chimeranet_latest.pt", {"epochs": 2})
    
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights"), exist_ok=True)
    import torch
    torch.save(ai_strategy.model.state_dict(), os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "chimeranet_latest.pt"))

if __name__ == "__main__":
    main()
