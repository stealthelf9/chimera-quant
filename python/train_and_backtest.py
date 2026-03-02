import os
import sys
import zipfile
import argparse
from datetime import datetime
import numpy as np

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
    extracted_dir = os.path.join(data_dir, "extracted")
    
    dbn_files = []
    if os.path.exists(extracted_dir):
        dbn_files = [os.path.join(extracted_dir, f) for f in os.listdir(extracted_dir) if f.endswith(".dbn.zst")]
        
    if not dbn_files:
        print("No extracted dbn.zst files found. Inspecting ZIPs...")
        zip_files = [f for f in os.listdir(data_dir) if f.endswith(".zip")]
        if not zip_files:
            print("No Databento ZIPs or extracted data found in data/.")
            return

        zip_path = os.path.join(data_dir, zip_files[0])
        print(f"Inspecting Archive: {zip_files[0]}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.endswith(".dbn.zst"):
                    print(f"Extracting {name} to data/extracted/...")
                    zf.extract(name, path=extracted_dir)
                    dbn_files.append(os.path.join(extracted_dir, name))

    if not dbn_files:
        print("No .dbn.zst files available.")
        return

    dbn_files.sort() # Ensure chronological order

    print("\n--- Loading Data to Zero-Copy Buffer ---")
    data_engine = chimera_core.MarketDataBuffer(10000000)
    for f in dbn_files:
        data_engine.load_dbn(f)
        
    print("\n--- Chronologically Aligning Numpy Array Memory ---")
    data_engine.sort_buffer()
        
    total_ticks = len(data_engine.get_buffer_view())
    print(f"Loaded {total_ticks} ticks natively mapped to NumPy.")
    
    view = data_engine.get_buffer_view()
    if 'instrument_id' in view.dtype.names:
        start_ns = date_to_nanos(args.start_date)
        end_ns = date_to_nanos(args.end_date) if args.end_date else 0xFFFFFFFFFFFFFFFF

        mask = (view['timestamp'] >= start_ns) & (view['timestamp'] <= end_ns)
        window_view = view[mask]
        
        if len(window_view) == 0:
            print(f"No data available in the requested timeframe: {args.start_date} to {args.end_date}")
            import sys
            sys.exit(0)
            
        instrument_ids, counts = np.unique(window_view['instrument_id'], return_counts=True)
        print(f"Found {len(instrument_ids)} unique instruments in timeframe.")
        
        if len(instrument_ids) >= 1:
            target_id = int(instrument_ids[np.argmax(counts)])
            print(f"Auto-selected most active instrument ID: {target_id} with {np.max(counts)} ticks in timeframe.")
            data_engine = data_engine.filter_by_instrument(target_id)
            total_ticks = len(data_engine.get_buffer_view())
            print(f"Filtered down to {total_ticks} ticks for isolated metric analysis.")

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

    print("\n--- Initializing Strategies ---")
    strategies_list = [s.strip().lower() for s in args.strategies.split(",")]
    signals = np.zeros(total_ticks, dtype=np.int32)
    
    if "ai" in strategies_list:
        train_size = int(total_ticks * 0.8)
        ai_strategy = AIStrategy(name="ChimeraNet_Alpha", params={"window_size": 30})
        
        if args.mode == "train":
            ai_strategy.buffer = data_engine.slice(0, train_size)
            ai_strategy.train(epochs=1, batch_size=16384)
            # Save weights
            weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
            os.makedirs(weights_dir, exist_ok=True)
            import torch
            torch.save(ai_strategy.model.state_dict(), os.path.join(weights_dir, "chimeranet_latest.pt"))
            np.save(os.path.join(weights_dir, "feature_mean.npy"), ai_strategy.feature_mean)
            np.save(os.path.join(weights_dir, "feature_std.npy"), ai_strategy.feature_std)
            print("Training finished and weights/scalers saved natively.")
            return

        elif args.mode == "backtest":
            weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
            weights_path = os.path.join(weights_dir, "chimeranet_latest.pt")
            mean_path = os.path.join(weights_dir, "feature_mean.npy")
            std_path = os.path.join(weights_dir, "feature_std.npy")
            if os.path.exists(weights_path):
                import torch
                ai_strategy.model.load_state_dict(torch.load(weights_path, map_location=ai_strategy.device, weights_only=True))
            if os.path.exists(mean_path) and os.path.exists(std_path):
                ai_strategy.feature_mean = np.load(mean_path)
                ai_strategy.feature_std = np.load(std_path)
            ai_strategy.model.eval()

            print("\n--- Generating Predictions for Validation Set ---")
            
            view = data_engine.get_buffer_view()
            start_ns = date_to_nanos(args.start_date)
            end_ns = date_to_nanos(args.end_date) if args.end_date else 0xFFFFFFFFFFFFFFFF
            
            mask = (view['timestamp'] >= start_ns) & (view['timestamp'] <= end_ns)
            valid_indices = np.where(mask)[0]
            
            if len(valid_indices) == 0:
                print("No data in requested timeframe.")
                import sys
                sys.exit(0)
                
            start_eval_idx = max(ai_strategy.window_size, valid_indices[0])
            end_eval_idx = valid_indices[-1] + 1
            
            # Extract features fully padded
            features = np.column_stack((
                view['open'],
                view['high'],
                view['low'],
                view['close'],
                view['volume']
            )).astype(np.float32)

            if hasattr(ai_strategy, 'feature_mean') and hasattr(ai_strategy, 'feature_std'):
                features = (features - ai_strategy.feature_mean) / ai_strategy.feature_std
                
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Generate Sliding Windows natively directly in NumPy
            X_all = np.lib.stride_tricks.sliding_window_view(
                features, (ai_strategy.window_size, 5)
            ).squeeze(axis=1)
            
            # Map index subset targets for PyTorch
            valid_X = X_all[start_eval_idx - ai_strategy.window_size : end_eval_idx - ai_strategy.window_size]
            
            if len(valid_X) == 0:
                print("Not enough contiguous data for AI evaluation window.")
                import sys
                sys.exit(0)
            
            # Use chunks so we don't overflow AMD ROCm LSTM max-sequence sizes in one pass
            chunk_size = 16384
            all_preds = []
            
            with torch.no_grad():
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    X_tensor = torch.tensor(valid_X.copy(), dtype=torch.float32).to(ai_strategy.device)
                
                # Execute batched dataset forwards in chunks
                for i in range(0, len(X_tensor), chunk_size):
                    chunk = X_tensor[i : i + chunk_size]
                    preds = ai_strategy.model(chunk).cpu().numpy().flatten()
                    all_preds.append(preds)
            
            if all_preds:
                predictions = np.concatenate(all_preds)
            else:
                predictions = np.array([])
            
            # Assign executed logic natively to signal arrays
            buy_mask = predictions > 0.00
            sell_mask = predictions < -0.01
            
            signals[start_eval_idx:end_eval_idx][buy_mask] = 1
            signals[start_eval_idx:end_eval_idx][sell_mask] = -1
            
            print(f"Generated {np.sum(buy_mask)} BUYS and {np.sum(sell_mask)} SELLS")

    else:
        # Standard Indicator Logic (No PyTorch Initialization)
        print("\n--- Generating Standard Indicator Predictions (Vectorized) ---")
        if "rsi" in strategies_list:
            from python.strategies.indicators import Indicators
            rsi_array = Indicators.rsi(data_engine, timeperiod=14)
            if len(rsi_array) > 0:
                valid_idx = ~np.isnan(rsi_array)
                signals[valid_idx & (rsi_array < 30)] = 1
                signals[valid_idx & (rsi_array > 70)] = -1
                
        elif "macd" in strategies_list:
            from python.strategies.indicators import Indicators
            macd, macdsignal, macdhist = Indicators.macd(data_engine)
            if len(macd) > 0:
                valid_idx = ~np.isnan(macd) & ~np.isnan(macdsignal)
                signals[valid_idx & (macd > macdsignal)] = 1
                signals[valid_idx & (macd < macdsignal)] = -1

    if args.mode == "backtest":
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
        print(f"Total Trades: {stats.total_trades}")

        print("\nCaching Results to SQLite...")
        cache = EvaluationCache()
        cache.log_backtest("ChimeraNet/RSI", {"window_size": 30}, 0, 0, stats.net_profit_usd, stats.max_drawdown)

if __name__ == "__main__":
    main()
