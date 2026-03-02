import os
from contextlib import asynccontextmanager
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Top 10 Mega-Caps constraint
MEGA_CAPS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "LLY", "V"]

class TradeExecutor:
    """
    Handles live Execution Domain rules via Alpaca Trading API (Paper Mode).
    """
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        print(f"Initializing Alpaca Trading Client (Paper={paper})...")
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def execute_signal(self, symbol: str, signal: str, quantity: float, current_price: float, slippage_tolerance: float = 0.005):
        """
        Executes a live trading constraint:
        1. Only allowed on Top 10 Mega Caps.
        2. Strictly Limit Orders with a widened slippage threshold.
        """
        if symbol not in MEGA_CAPS:
            print(f"[REJECTED] {symbol} is not inside the MEGA_CAPS allowable pipeline.")
            return False

        if signal not in ["BUY", "SELL"]:
            print(f"[ERROR] Unknown signal {signal}")
            return False

        # Apply custom slippage tolerance calculation
        if signal == "BUY":
            limit_price = round(current_price * (1.0 + slippage_tolerance), 2)
            side = OrderSide.BUY
        else: # SELL
            limit_price = round(current_price * (1.0 - slippage_tolerance), 2)
            side = OrderSide.SELL

        print(f"[{symbol}] Firing {signal} Order for {quantity} shares. Limit Price Set: {limit_price} (Current: {current_price})")

        request = LimitOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price
        )

        try:
            order = self.client.submit_order(request)
            print(f"[SUCCESS] Order ID {order.id} submitted for {symbol}.")
            return order
        except Exception as e:
            print(f"[FAILED] Trade execution threw an exception: {e}")
            return None

if __name__ == "__main__":
    # Test Dry-Run Execution Pipeline (Note: Uses the provided free tier keys)
    API_KEY = "PKOO9VTOEYVG4EBIIXXA"
    SECRET_KEY = "OLqMI9JtYfHbruwyzFDSMs9cmAdug5kAwFZJjyrJ"
    
    executor = TradeExecutor(api_key=API_KEY, secret_key=SECRET_KEY, paper=True)
    
    # Test Constraint 1: Should Reject
    print("Testing Security Constraint Logic...")
    executor.execute_signal(symbol="GME", signal="BUY", quantity=10, current_price=25.0, slippage_tolerance=0.01)
    
    # Test Constraint 2: Should Process (But we may hit Alpaca Market Closed logic depending on time)
    print("\nTesting Valid Pipeline Execution...")
    executor.execute_signal(symbol="NVDA", signal="BUY", quantity=1, current_price=105.0, slippage_tolerance=0.01)
