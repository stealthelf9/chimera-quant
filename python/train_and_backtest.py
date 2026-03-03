import os
import sys
import zipfile
import argparse
from datetime import datetime
import numpy as np
import databento as db

# Ensure CMake build output is in PYTHONPATH for the chimera_core C++ module
BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")
sys.path.append(BUILD_DIR)

import chimera_core
from python.strategies.model import AIStrategy
from python.storage.cache import EvaluationCache

def parse_args():
    parser = argparse.ArgumentParser(description="Chimera Quant Backtest & Training CLI")
    parser.add_argument("--mode", choices=["train", "backtest", "live"], default="backtest", help="Execution mode")
    parser.add_argument("--dataset", type=str, default="", help="Subfolder inside the data directory (e.g., 'equs' or 'itch'). Leave blank for root data folder.")
    parser.add_argument("--model-name", type=str, default="UNIVERSAL", help="Custom name for saving/loading model weights and scalers.")
    parser.add_argument("--resume", action="store_true", help="Load existing weights before training to resume a session.")
    parser.add_argument("--strategies", type=str, default="ai", help="Comma separated list of strategies")
    parser.add_argument("--timeframe", type=str, default="1m", help="Resampling timeframe (e.g., 1m, 5m, 1h, 1d)")
    parser.add_argument("--start-date", type=str, default="", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="", help="End date YYYY-MM-DD")
    parser.add_argument("--symbols", type=str, default="ALL", help="Comma separated symbols or ALL")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--position-size", type=float, default=0.1, help="Position size (0.1 = 10 percent of portfolio)")
    parser.add_argument("--slippage", type=float, default=0.005, help="Slippage penalty")
    parser.add_argument("--commission", type=str, default="2.5", help="Commission per trade (e.g., '1.5' for $1.5 flat, or '1.5%%' for 1.5%% percentage default)")
    parser.add_argument("--stop-loss", type=float, default=0.05, help="Stop loss percentage (e.g., 0.05 for 5%%)")
    parser.add_argument("--take-profit", type=float, default=0.10, help="Take profit percentage (e.g., 0.10 for 10%%)")
    parser.add_argument("--shorting", action="store_true", help="Enable short selling")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
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
    
    if args.dataset:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", args.dataset)
    else:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    dbn_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".dbn.zst")]
        
    if not dbn_files:
        print("No .dbn.zst files found in data/. Inspecting ZIPs...")
        zip_files = [f for f in os.listdir(data_dir) if f.endswith(".zip")]
        if not zip_files:
            print("No Databento ZIPs or extracted data found in data/.")
            return

        zip_path = os.path.join(data_dir, zip_files[0])
        print(f"Inspecting Archive: {zip_files[0]}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.endswith(".dbn.zst"):
                    print(f"Extracting {name} to data/...")
                    zf.extract(name, path=data_dir)
                    dbn_files.append(os.path.join(data_dir, name))

    if not dbn_files:
        print("No .dbn.zst files available.")
        return

    dbn_files.sort() # Ensure chronological order

    print("\n--- Loading Data to Zero-Copy Buffer ---")
    data_engine = chimera_core.MarketDataBuffer(10000000)
    
    print("Extracting symbology mappings from Databento...")
    store = db.DBNStore.from_file(dbn_files[0])
    ticker_map = {}
    for ticker, mappings in store.mappings.items():
        if mappings:
            ticker_map[ticker.upper()] = int(mappings[0]["symbol"])
            
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
        
        if args.symbols != "ALL":
            target_tickers = [s.strip().upper() for s in args.symbols.split(",")]
            target_ids_to_process = []
            for t in target_tickers:
                if t in ticker_map:
                    target_ids_to_process.append(ticker_map[t])
                else:
                    print(f"Warning: Ticker {t} not found in DB symbology.")
                    
            if not target_ids_to_process:
                print("No valid target instruments found to filter. Exiting.")
                return
            print(f"Targeting {len(target_ids_to_process)} instruments.")
        else:
            target_ids_to_process = instrument_ids.tolist()
            print(f"Targeting ALL {len(target_ids_to_process)} instruments dynamically.")
            
        total_ticks = len(data_engine.get_buffer_view())

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
        
        file_prefix = args.model_name
        
        if args.mode == "train":
            if args.resume:
                print(f"\n--- Resuming from previous weights: {file_prefix} ---")
                weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
                weights_path = os.path.join(weights_dir, f"chimeranet_{file_prefix}.pt")
                mean_path = os.path.join(weights_dir, f"feature_mean_{file_prefix}.npy")
                std_path = os.path.join(weights_dir, f"feature_std_{file_prefix}.npy")
                if os.path.exists(weights_path):
                    import torch
                    state_dict = torch.load(weights_path, map_location=ai_strategy.device, weights_only=True)
                    clean_dict = {}
                    for key, value in state_dict.items():
                        clean_key = key.replace("_orig_mod.", "")
                        clean_dict[clean_key] = value
                    
                    if hasattr(ai_strategy.model, '_orig_mod'):
                        ai_strategy.model._orig_mod.load_state_dict(clean_dict)
                    else:
                        ai_strategy.model.load_state_dict(clean_dict)
                    print(f"Loaded existing weights from {weights_path}")
                else:
                    print("Could not find weights file. Starting fresh.")
                    
                if os.path.exists(mean_path) and os.path.exists(std_path):
                    ai_strategy.feature_mean = np.load(mean_path)
                    ai_strategy.feature_std = np.load(std_path)
                    print(f"Loaded existing scalers from {mean_path} and {std_path}")
                else:
                    print("Could not find scalers. Starting fresh.")
                    
            # --- Global Scaler Calculation Phase ---
            print("\n--- Calculating Global Feature Scalers ---")
            all_features = []
            for t_id in target_ids_to_process:
                sub_engine = data_engine.filter_by_instrument(t_id)
                train_size = int(len(sub_engine.get_buffer_view()) * 0.8)
                if train_size < 100:
                    continue
                    
                view = sub_engine.slice(0, train_size).get_buffer_view()
                closes = view['close']
                returns = np.zeros_like(closes, dtype=np.float32)
                with np.errstate(divide='ignore', invalid='ignore'):
                    returns[1:] = np.where(closes[:-1] < 1e-8, 0.0, ((closes[1:] - closes[:-1]) / closes[:-1]) * 100.0)
                    
                from python.strategies.indicators import Indicators
                rsi = Indicators.rsi(sub_engine.slice(0, train_size), timeperiod=14)
                _, _, macdhist = Indicators.macd(sub_engine.slice(0, train_size))
                
                features = np.column_stack((returns, view['volume'], rsi, macdhist)).astype(np.float32)
                all_features.append(features)
                
            if all_features:
                global_features = np.concatenate(all_features, axis=0)
                ai_strategy.feature_mean = np.nanmean(global_features, axis=0, dtype=np.float64)
                ai_strategy.feature_std = np.nanstd(global_features, axis=0, dtype=np.float64)
                ai_strategy.feature_std[ai_strategy.feature_std == 0] = 1.0
                print("Global Scalers Computed.")
                
            total_assets = len(target_ids_to_process)
            import time
            global_start_time = time.time()
            
            try:
                for idx, t_id in enumerate(target_ids_to_process, start=1):
                    asset_start_time = time.time()
                    sub_engine = data_engine.filter_by_instrument(t_id)
                    train_size = int(len(sub_engine.get_buffer_view()) * 0.8)
                    if train_size < 100:
                        continue
                    ai_strategy.buffer = sub_engine.slice(0, train_size)
                    print(f"\n--- Training on Instrument ID: {t_id} [{idx}/{total_assets} Assets] ---")
                    ai_strategy.train(epochs=args.epochs, batch_size=args.batch_size)
                    
                    asset_elapsed = time.time() - asset_start_time
                    print(f"--- Asset {t_id} Completed in {asset_elapsed:.2f}s ---")
                    
                global_elapsed = time.time() - global_start_time
                print(f"\n--- All Assets Completed! Total Elapsed Time: {global_elapsed:.2f}s ---")
            except KeyboardInterrupt:
                print(f"\nTraining interrupted. Gracefully saving current weights to chimeranet_{file_prefix}.pt...")
            
            # Save weights
            weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
            os.makedirs(weights_dir, exist_ok=True)
            import torch
            torch.save(ai_strategy.model.state_dict(), os.path.join(weights_dir, f"chimeranet_{file_prefix}.pt"))
            np.save(os.path.join(weights_dir, f"feature_mean_{file_prefix}.npy"), ai_strategy.feature_mean)
            np.save(os.path.join(weights_dir, f"feature_std_{file_prefix}.npy"), ai_strategy.feature_std)
            print(f"Training finished and Universal weights/scalers saved natively as {file_prefix}.")
            return

        elif args.mode == "backtest":
            print("\n=========================================")
            print("WARNING: Feature space has drastically changed to 4 Stationary Metrics (Returns, Volume, RSI, MACDHist).")
            print("Ensure you run --mode train again before running backtests or PyTorch LSTM Tensor dimensions will mismatch!")
            print("=========================================\n")
            
            weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
            weights_path = os.path.join(weights_dir, f"chimeranet_{file_prefix}.pt")
            mean_path = os.path.join(weights_dir, f"feature_mean_{file_prefix}.npy")
            std_path = os.path.join(weights_dir, f"feature_std_{file_prefix}.npy")
            if os.path.exists(weights_path):
                import torch
                state_dict = torch.load(weights_path, map_location=ai_strategy.device, weights_only=True)
                
                # Clean out the compiler prefixes dynamically
                clean_dict = {}
                for key, value in state_dict.items():
                    clean_key = key.replace("_orig_mod.", "")
                    clean_dict[clean_key] = value
                
                # Load the perfectly clean weights
                if hasattr(ai_strategy.model, '_orig_mod'):
                    ai_strategy.model._orig_mod.load_state_dict(clean_dict)
                else:
                    ai_strategy.model.load_state_dict(clean_dict)
                
            if os.path.exists(mean_path) and os.path.exists(std_path):
                ai_strategy.feature_mean = np.load(mean_path)
                ai_strategy.feature_std = np.load(std_path)
            ai_strategy.model.eval()

            print("\n--- Generating Predictions for Validation Set ---")
            
            view = data_engine.get_buffer_view()
            start_ns = date_to_nanos(args.start_date)
            end_ns = date_to_nanos(args.end_date) if args.end_date else 0xFFFFFFFFFFFFFFFF
            
            for t_id in target_ids_to_process:
                # Isolate sub-engine to prevent MSFT->AAPL diff spiking
                sub_engine = data_engine.filter_by_instrument(t_id)
                sub_view = sub_engine.get_buffer_view()
                if len(sub_view) < ai_strategy.window_size + 10:
                    continue
                
                mask = (sub_view['timestamp'] >= start_ns) & (sub_view['timestamp'] <= end_ns)
                valid_indices = np.where(mask)[0]
                if len(valid_indices) == 0:
                    continue
                    
                start_eval_idx = max(ai_strategy.window_size, valid_indices[0])
                end_eval_idx = valid_indices[-1] + 1
                
                # Calculate stationary features
                closes = sub_view['close']
                returns = np.zeros_like(closes, dtype=np.float32)
                with np.errstate(divide='ignore', invalid='ignore'):
                    returns[1:] = np.where(closes[:-1] < 1e-8, 0.0, ((closes[1:] - closes[:-1]) / closes[:-1]) * 100.0)
                    
                from python.strategies.indicators import Indicators
                rsi = Indicators.rsi(sub_engine, timeperiod=14)
                _, _, macdhist = Indicators.macd(sub_engine)

                # Extract features fully padded
                features = np.column_stack((
                    returns,
                    sub_view['volume'],
                    rsi,
                    macdhist
                )).astype(np.float32)

                if hasattr(ai_strategy, 'feature_mean') and hasattr(ai_strategy, 'feature_std'):
                    features = (features - ai_strategy.feature_mean) / ai_strategy.feature_std
                    
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

                # Generate Sliding Windows natively directly in NumPy
                X_all = np.lib.stride_tricks.sliding_window_view(
                    features, (ai_strategy.window_size, 4)
                ).squeeze(axis=1)
                
                # Map index subset targets for PyTorch
                valid_X = X_all[start_eval_idx - ai_strategy.window_size : end_eval_idx - ai_strategy.window_size]
                
                if len(valid_X) == 0:
                    continue
                
                chunk_size = 16384
                all_preds = []
                
                with torch.no_grad():
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        X_tensor = torch.tensor(valid_X.copy(), dtype=torch.float32).to(ai_strategy.device)
                    
                    for i in range(0, len(X_tensor), chunk_size):
                        chunk = X_tensor[i : i + chunk_size]
                        preds = ai_strategy.model(chunk).cpu().numpy().flatten()
                        all_preds.append(preds)
                
                if all_preds:
                    predictions = np.concatenate(all_preds)
                else:
                    predictions = np.array([])
                
                # Assign executed logic natively to localized sub arrays
                buy_mask = predictions > 0.05
                sell_mask = predictions < -0.30
                
                # We map the local sub_view start_eval_idx back directly to global view indices via timestamp mapping
                global_mask = (view['instrument_id'] == t_id) & (view['timestamp'] >= sub_view['timestamp'][start_eval_idx]) & (view['timestamp'] <= sub_view['timestamp'][end_eval_idx - 1])
                global_indices_mapped = np.where(global_mask)[0]
                
                # Ensure mapping lengths align exactly
                if len(global_indices_mapped) == len(predictions):
                    signals[global_indices_mapped[buy_mask]] = 1
                    signals[global_indices_mapped[sell_mask]] = -1
                
            print(f"Generated {np.sum(signals == 1)} BUYS and {np.sum(signals == -1)} SELLS")


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
        
        # Parse dynamic commission
        commission_str = str(args.commission).strip()
        commission_is_percentage = False
        commission_value = 0.0
        if "%" in commission_str:
            commission_is_percentage = True
            commission_value = float(commission_str.replace("%", "")) / 100.0
        else:
            commission_value = float(commission_str)
        
        stats = data_engine.run_backtest(
            initial_capital=args.capital,
            order_size=args.position_size,
            size_is_percentage=True,
            commission=commission_value,
            commission_is_percentage=commission_is_percentage,
            slippage_penalty=args.slippage,
            start_timestamp=start_ns,
            end_timestamp=end_ns,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit,
            allow_shorts=args.shorting,
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
