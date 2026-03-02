#pragma once
#include "types.h"
#include <cstdint>
#include <vector>

namespace chimera {

// Structure holding performance results to return directly to Python
struct BacktestStats {
  double net_profit_usd;
  double net_profit_pct;
  double win_rate;
  double max_drawdown;
  double sharpe_ratio;
  int total_trades;

  // Trade logs could be expanded natively here, but for now we'll
  // just return the summarized stats for execution efficiency.
};

// C++ implementation of the simulation loop
class BacktestSimulator {
public:
  BacktestSimulator(std::vector<OHLCV> &buffer_ref);

  // Run the native execution
  BacktestStats
  run(double initial_capital,
      double order_size, // Fixed dollar or % logic to be implemented
      bool size_is_percentage, double commission, bool commission_is_percentage,
      double slippage_penalty, // Example: 0.005 (50 bps)
      uint64_t start_timestamp, uint64_t end_timestamp, double stop_loss_pct,
      double take_profit_pct, bool allow_shorts,
      const std::vector<int> &signals // Example logic: 1 is BUY, -1 is SELL, 0
                                      // HOLD. Shape matches buffer.
  );

private:
  std::vector<OHLCV> &data;
};

} // namespace chimera
