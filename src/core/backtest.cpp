#include "backtest.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map> // Ensure unordered_map is included for PositionState

namespace chimera {

BacktestSimulator::BacktestSimulator(std::vector<OHLCV> &buffer_ref)
    : data(buffer_ref) {}

BacktestStats
BacktestSimulator::run(double initial_capital, double order_size,
                       bool size_is_percentage, double commission,
                       bool commission_is_percentage, double slippage_penalty,
                       uint64_t start_timestamp, uint64_t end_timestamp,
                       double stop_loss_pct, double take_profit_pct,
                       bool allow_shorts, const std::vector<int> &signals) {

  BacktestStats stats{};
  if (data.empty() || signals.size() != data.size()) {
    std::cerr << "Backtest aborted: Buffer empty or signal mismatch."
              << std::endl;
    return stats;
  }

  double equity = initial_capital;
  double peak_equity = initial_capital;
  double max_dd = 0.0;

  int total_trades = 0;
  int winning_trades = 0;

  std::unordered_map<uint32_t, PositionState> positions;
  std::unordered_map<uint32_t, double> latest_prices;

  std::vector<double> daily_returns;
  double last_equity = equity;

  auto start_it = std::lower_bound(
      data.begin(), data.end(), start_timestamp,
      [](const OHLCV &t, uint64_t ts) { return t.timestamp < ts; });
  size_t start_idx = std::distance(data.begin(), start_it);

  size_t last_valid_idx = start_idx < data.size() ? start_idx : 0;

  for (size_t i = start_idx; i < data.size(); ++i) {
    const auto &tick = data[i];

    if (tick.timestamp > end_timestamp)
      break;

    last_valid_idx = i;
    latest_prices[tick.instrument_id] = tick.close;

    int signal = (i > 0) ? signals[i - 1] : 0;
    auto &pos = positions[tick.instrument_id];

    // --- Risk Management (SL / TP) ---
    bool risk_triggered = false;
    double exit_price = 0.0;

    if (pos.type == 1 && pos.shares > 0) {
      double sl_price = pos.entry_price * (1.0 - stop_loss_pct);
      double tp_price = pos.entry_price * (1.0 + take_profit_pct);
      if (tick.low <= sl_price) {
        exit_price = sl_price * (1.0 - slippage_penalty);
        risk_triggered = true;
      } else if (tick.high >= tp_price) {
        exit_price = tp_price * (1.0 - slippage_penalty);
        risk_triggered = true;
      }
    } else if (pos.type == -1 && pos.shares > 0) {
      double sl_price = pos.entry_price * (1.0 + stop_loss_pct);
      double tp_price = pos.entry_price * (1.0 - take_profit_pct);
      if (tick.high >= sl_price) {
        exit_price = sl_price * (1.0 + slippage_penalty);
        risk_triggered = true;
      } else if (tick.low <= tp_price) {
        exit_price = tp_price * (1.0 + slippage_penalty);
        risk_triggered = true;
      }
    }

    if (risk_triggered) {
      double trade_value = pos.shares * exit_price;
      double fee =
          commission_is_percentage ? (trade_value * commission) : commission;

      double trade_profit = 0;
      if (pos.type == 1) {
        equity += (trade_value - fee);
        trade_profit = (exit_price - pos.entry_price) * pos.shares - fee;
      } else if (pos.type == -1) {
        trade_profit = (pos.entry_price - exit_price) * pos.shares - fee;
        equity += trade_profit;
      }

      if (trade_profit > 0)
        winning_trades++;
      total_trades++;

      pos.shares = 0;
      pos.entry_price = 0;
      pos.type = 0;
    }

    // --- Signal Handling ---
    if (!risk_triggered && signal != 0) {
      if (signal == -1) {
        // Close Long position (Flip Constraint: Flat first)
        if (pos.type == 1 && pos.shares > 0) {
          exit_price = tick.open * (1.0 - slippage_penalty);
          double trade_value = pos.shares * exit_price;
          double fee = commission_is_percentage ? (trade_value * commission)
                                                : commission;

          equity += (trade_value - fee);
          double trade_profit =
              (exit_price - pos.entry_price) * pos.shares - fee;
          if (trade_profit > 0)
            winning_trades++;
          total_trades++;

          pos.shares = 0;
          pos.entry_price = 0;
          pos.type = 0;
        }
        // Open Short position (If flat)
        else if (pos.type == 0 && allow_shorts) {
          double bid_price = tick.open * (1.0 - slippage_penalty);
          double current_marked_equity = equity;
          for (const auto &pair : positions) {
            const auto &p = pair.second;
            if (p.shares > 0) {
              double lp = latest_prices[pair.first];
              if (p.type == 1)
                current_marked_equity += p.shares * lp;
              else if (p.type == -1)
                current_marked_equity += (p.entry_price - lp) * p.shares;
            }
          }
          double investment = size_is_percentage
                                  ? (current_marked_equity * order_size)
                                  : order_size;
          investment = std::min(investment,
                                equity); // can't invest more cash than we have
          if (investment > 0) {
            double fee = commission_is_percentage ? (investment * commission)
                                                  : commission;
            investment -= fee;
            if (investment > 0) {
              pos.entry_price = bid_price;
              pos.shares = investment / bid_price;
              pos.type = -1;
              equity -= fee;
            }
          }
        }
      } else if (signal == 1) {
        // Close Short position (Flip Constraint: Flat first)
        if (pos.type == -1 && pos.shares > 0) {
          exit_price = tick.open * (1.0 + slippage_penalty);
          double trade_value = pos.shares * exit_price;
          double fee = commission_is_percentage ? (trade_value * commission)
                                                : commission;

          double trade_profit =
              (pos.entry_price - exit_price) * pos.shares - fee;
          equity += trade_profit;

          if (trade_profit > 0)
            winning_trades++;
          total_trades++;

          pos.shares = 0;
          pos.entry_price = 0;
          pos.type = 0;
        }
        // Open Long position (if flat)
        else if (pos.type == 0) {
          double ask_price = tick.open * (1.0 + slippage_penalty);
          double current_marked_equity = equity;
          for (const auto &pair : positions) {
            const auto &p = pair.second;
            if (p.shares > 0) {
              double lp = latest_prices[pair.first];
              if (p.type == 1)
                current_marked_equity += p.shares * lp;
              else if (p.type == -1)
                current_marked_equity += (p.entry_price - lp) * p.shares;
            }
          }
          double investment = size_is_percentage
                                  ? (current_marked_equity * order_size)
                                  : order_size;
          investment = std::min(investment, equity);
          if (investment > 0) {
            double fee = commission_is_percentage ? (investment * commission)
                                                  : commission;
            investment -= fee;
            if (investment > 0) {
              pos.entry_price = ask_price;
              pos.shares = investment / ask_price;
              pos.type = 1;
              equity -= (investment + fee);
            }
          }
        }
      }
    }

    // Daily/Periodic Returns Tracking for Sharpe
    double current_marked_equity = equity;
    for (const auto &pair : positions) {
      const auto &p = pair.second;
      if (p.shares > 0) {
        double lp = latest_prices[pair.first];
        if (p.type == 1) {
          current_marked_equity += p.shares * lp;
        } else if (p.type == -1) {
          current_marked_equity += (p.entry_price - lp) * p.shares;
        }
      }
    }

    peak_equity = std::max(peak_equity, current_marked_equity);
    double draw_down = (peak_equity - current_marked_equity) / peak_equity;
    max_dd = std::max(max_dd, draw_down);

    double period_return = (current_marked_equity - last_equity) / last_equity;
    daily_returns.push_back(period_return);
    last_equity = current_marked_equity;
  }

  // Close remaining positions at the end of simulation
  for (auto &pair : positions) {
    auto &pos = pair.second;
    if (pos.shares > 0) {
      // Find the last actual tick for this specific instrument id
      // Since data is chronological, latest_prices is the final valid tick we
      // have for it
      double exit_price = latest_prices[pair.first];
      if (pos.type == 1) {
        exit_price *= (1.0 - slippage_penalty);
        double trade_value = pos.shares * exit_price;
        double fee =
            commission_is_percentage ? (trade_value * commission) : commission;
        equity += (trade_value - fee);

        double trade_profit = (exit_price - pos.entry_price) * pos.shares - fee;
        if (trade_profit > 0)
          winning_trades++;
        total_trades++;
      } else if (pos.type == -1) {
        exit_price *= (1.0 + slippage_penalty);
        double trade_value = pos.shares * exit_price;
        double fee =
            commission_is_percentage ? (trade_value * commission) : commission;

        double trade_profit = (pos.entry_price - exit_price) * pos.shares - fee;
        equity += trade_profit;

        if (trade_profit > 0)
          winning_trades++;
        total_trades++;
      }
      pos.shares = 0;
    }
  }

  // Net Profit Calculations
  stats.net_profit_usd = equity - initial_capital;
  stats.net_profit_pct = (stats.net_profit_usd / initial_capital) * 100.0;
  stats.win_rate = (total_trades > 0)
                       ? ((double)winning_trades / total_trades) * 100.0
                       : 0.0;
  stats.max_drawdown = max_dd * 100.0;
  stats.total_trades = total_trades;

  // Standard Deviation of Returns Native Compute
  if (!daily_returns.empty()) {
    double sum = 0;
    for (double r : daily_returns)
      sum += r;
    double mean = sum / daily_returns.size();

    double varianceSum = 0;
    for (double r : daily_returns)
      varianceSum += (r - mean) * (r - mean);
    double std_dev = std::sqrt(varianceSum / daily_returns.size());

    // Annualized Sharpe using generic 252 (Days) * 390 (Minutes) roughly
    if (std_dev > 0) {
      stats.sharpe_ratio = (mean / std_dev) * std::sqrt(252 * 390);
    } else {
      stats.sharpe_ratio = 0;
    }
  }

  return stats;
}

} // namespace chimera
