python python/train_and_backtest.py \
    --mode train \
    --dataset equs \
    --model-name "S&P_1min_V1" \
    --start-date 2023-01-01 \
    --end-date 2024-12-31 \
    --symbols ALL \
    --timeframe 1m \
    --epochs 50 \
    --batch-size 8192