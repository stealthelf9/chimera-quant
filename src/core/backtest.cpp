#include "backtest.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace chimera {

BacktestSimulator::BacktestSimulator(std::vector<OHLCV> &buffer_ref)
    : data(buffer_ref) {}

BacktestStats BacktestSimulator::run(double initial_capital, double order_size,
                                     bool size_is_percentage, double commission,
                                     bool commission_is_percentage,
                                     double slippage_penalty,
                                     uint64_t start_timestamp,
                                     uint64_t end_timestamp,
                                     const std::vector<int> &signals) {

  BacktestStats stats = {0};
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

  double position_shares = 0;
  double entry_price = 0;

  std::vector<double> daily_returns;
  double last_equity = equity;

  auto start_it = std::lower_bound(
      data.begin(), data.end(), start_timestamp,
      [](const OHLCV &t, uint64_t ts) { return t.timestamp < ts; });
  size_t start_idx = std::distance(data.begin(), start_it);

  for (size_t i = start_idx; i < data.size(); ++i) {
    const auto &tick = data[i];

    if (tick.timestamp > end_timestamp)
      break;

    int signal = (i > 0) ? signals[i - 1] : 0;

    // Close position
    if (signal == -1 && position_shares > 0) {
      double exit_price = tick.open * (1.0 - slippage_penalty);
      double trade_value = position_shares * exit_price;
      double fee =
          commission_is_percentage ? (trade_value * commission) : commission;

      equity += (trade_value - fee);

      double trade_profit = (exit_price - entry_price) * position_shares - fee;
      if (trade_profit > 0)
        winning_trades++;
      total_trades++;

      position_shares = 0;
      entry_price = 0;
    }
    // Open position
    else if (signal == 1 && position_shares == 0) {
      double ask_price = tick.open * (1.0 + slippage_penalty);
      double investment = size_is_percentage ? (equity * order_size)
                                             : std::min(order_size, equity);

      double fee =
          commission_is_percentage ? (investment * commission) : commission;
      investment -= fee;

      entry_price = ask_price;
      position_shares = investment / ask_price;
      equity -= (investment + fee);
    }

    // Daily/Periodic Returns Tracking for Sharpe
    // Using a basic heuristic tracking per step for testing
    double current_marked_equity = equity + (position_shares * tick.close);

    peak_equity = std::max(peak_equity, current_marked_equity);
    double draw_down = (peak_equity - current_marked_equity) / peak_equity;
    max_dd = std::max(max_dd, draw_down);

    double period_return = (current_marked_equity - last_equity) / last_equity;
    daily_returns.push_back(period_return);
    last_equity = current_marked_equity;
  }

  // Close remaining position at the end of simulation
  if (position_shares > 0) {
    double exit_price = data.back().close * (1.0 - slippage_penalty);
    double trade_value = position_shares * exit_price;
    double fee =
        commission_is_percentage ? (trade_value * commission) : commission;
    equity += (trade_value - fee);

    double trade_profit = (exit_price - entry_price) * position_shares - fee;
    if (trade_profit > 0)
      winning_trades++;
    total_trades++;
  }

  // Net Profit Calculations
  stats.net_profit_usd = equity - initial_capital;
  stats.net_profit_pct = (stats.net_profit_usd / initial_capital) * 100.0;
  stats.win_rate = (total_trades > 0)
                       ? ((double)winning_trades / total_trades) * 100.0
                       : 0.0;
  stats.max_drawdown = max_dd * 100.0;

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
    // proxying
    if (std_dev > 0) {
      stats.sharpe_ratio = (mean / std_dev) * std::sqrt(252 * 390);
    } else {
      stats.sharpe_ratio = 0;
    }
  }

  return stats;
}

} // namespace chimera
