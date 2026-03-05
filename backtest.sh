#!/bin/bash

python python/train_and_backtest.py \
    --mode backtest \
    --dataset equs \
    --model-name "S&P_1min_V1" \
    --start-date 2025-01-01 \
    --end-date 2026-02-28 \
    --strategies ai \
    --timeframe 1m \
    --capital 10000 \
    --position-size 0.05 \
    --commission 0.00 \
    --slippage 0.0001 \
    --stop-loss 0.003 \
    --take-profit 0.006 \
    --shorting \
    --symbols ALL