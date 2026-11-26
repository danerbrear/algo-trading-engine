"""Momentum strategy that sells put credit spreads when short-term moving averages accelerate."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional, Sequence
import pandas as pd

from algo_trading_engine._internal.util.market_clock import label_position, market_index
from algo_trading_engine._internal.util.date_formats import DateFormats
from algo_trading_engine.data_retriever import DataRetriever
from algo_trading_engine.strategy import Strategy
from algo_trading_engine.vo.universe import PostLoadCallback, UniverseSet
from algo_trading_engine.indicators.sma_indicator import SMAIndicator
from algo_trading_engine.vo import Position, create_position
from algo_trading_engine.enums import OptionType, StrategyType
from algo_trading_engine.vo import Option, TreasuryRates
from algo_trading_engine.dto import ExpirationRangeDTO, OptionContractDTO, OptionsChainDTO, StrikeRangeDTO
from algo_trading_engine.vo import StrikePrice
from algo_trading_engine.logging import get_logger
from algo_trading_engine.options_helpers import OptionsRetrieverHelper
from decimal import Decimal


class VelocitySignalMomentumStrategy(Strategy):
    """
    A momentum strategy to trade credit spreads in order to capitalize on
    the upward or downward trends.
    """

    # Configurable holding period in trading days
    holding_period = 4
    universe_set: UniverseSet = ["SPY"]

    def __init__(
        self,
        get_contract_list_for_date: Optional[Callable] = None,
        get_option_bar: Optional[Callable] = None,
        get_options_chain: Optional[Callable] = None,
        stop_loss: Optional[float] = None,
        profit_target: Optional[float] = None,
    ):
        super().__init__(stop_loss=stop_loss, profit_target=profit_target)

        if get_contract_list_for_date is not None:
            self.get_contract_list_for_date = get_contract_list_for_date
        if get_option_bar is not None:
            self.get_option_bar = get_option_bar
        if get_options_chain is not None:
            self.get_options_chain = get_options_chain
        self.options_data: dict[str, OptionsChainDTO] = {}

        # Track position entries for plotting
        self._position_entries = []

        self.add_indicator(SMAIndicator(period=15))
        self.add_indicator(SMAIndicator(period=30))

    @staticmethod
    def _enrich_velocity_frame(symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        del symbol
        enriched = data.copy()
        if enriched.empty:
            return enriched
        enriched["SMA_15"] = enriched["Close"].rolling(window=15).mean()
        enriched["SMA_30"] = enriched["Close"].rolling(window=30).mean()
        enriched["MA_Velocity_15_30"] = enriched["SMA_15"] / enriched["SMA_30"]
        enriched["Velocity_Changes"] = enriched["MA_Velocity_15_30"].diff()
        return enriched

    @staticmethod
    def universe_post_load(retriever: DataRetriever) -> PostLoadCallback:
        del retriever
        return VelocitySignalMomentumStrategy._enrich_velocity_frame

    def set_universe(self, universe, treasury_data: Optional[TreasuryRates] = None) -> None:
        super().set_universe(universe, treasury_data)
        self._position_entries = []

    def on_new_date(
        self,
        date: datetime,
        positions: tuple["Position", ...],
        add_position: Callable[["Position"], None],
        remove_position: Strategy.RemovePositionCallback,
    ):
        super().on_new_date(date, positions, add_position, remove_position)

        if self.universe is None:
            return
        for symbol in self.universe.symbols:
            if not self.universe.has_bar(symbol, date):
                continue
            symbol_positions = tuple(p for p in positions if p.symbol == symbol)
            if len(symbol_positions) == 0:
                self._try_open_position(date, add_position, symbol)
            self._try_close_positions(date, symbol_positions, remove_position, symbol)

    def on_end(
        self,
        positions: Sequence["Position"],
        remove_position: Strategy.RemovePositionCallback,
        date: datetime,
    ):
        """End-of-run hook; plotting is handled by the engine / show_plot callers."""
        del positions, remove_position, date

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the data for the Velocity Signal Momentum Strategy.

        This strategy requires:
        - Basic OHLCV data
        - Moving averages (SMA_15, SMA_30) for velocity calculation
        - Velocity changes data

        Args:
            data: DataFrame with market data and features

        Returns:
            bool: True if data is valid for this strategy, False otherwise
        """
        get_logger().info("\n🔍 Validating data for Velocity Signal Momentum Strategy...")
        get_logger().info(f"   Data shape: {data.shape}")

        # Check if the data has the required columns for velocity momentum strategy
        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",  # Basic OHLCV data
        ]

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            get_logger().warning(f"⚠️  Warning: Missing required columns: {missing_columns}")
            get_logger().info(f"   Available columns: {list(data.columns)}")
            return False

        get_logger().info("✅ All required columns present")

        try:
            market_index(data)
        except (TypeError, ValueError) as error:
            get_logger().error(f"❌ Error: {error}")
            return False

        # Check if we have enough data for moving averages (need at least 30 days for SMA 30)
        if len(data) < 30:
            get_logger().warning(
                f"⚠️  Warning: Not enough data for velocity analysis. Need at least 30 days, got {len(data)}"
            )
            return False

        # Check for gaps in the data (missing trading days)
        if len(data) > 1:
            date_range = pd.bdate_range(start=data.index.min(), end=data.index.max())
            expected_business_days = len(date_range)
            actual_trading_days = len(data)
            if actual_trading_days < expected_business_days * 0.9:  # Allow for some holidays
                get_logger().warning(
                    f"⚠️  Warning: Data may have gaps. Expected ~{expected_business_days} business days, got {actual_trading_days}"
                )

        get_logger().info("✅ Data validation complete for Velocity Signal Momentum Strategy")
        get_logger().info(f"   Final data shape: {data.shape}")
        get_logger().info(f"   Date range: {data.index.min()} to {data.index.max()}")
        get_logger().info(f"   Trading days: {len(data)}")

        return True

    def _has_buy_signal(self, date: datetime, symbol: str) -> bool:
        """
        Check for buy signal using MA velocity (SMA 15/30) approach.

        If the data matches the following criteria for a buy signal, return True:
            - MA velocity (SMA 15/30) must increase (signal detected)
            - Price must increase over the trend period (3-60 days)
            - No significant reversals (>2% drop) during the trend
            - Trend must last at least 3 days
            - Trend must not exceed 60 days
        """
        try:
            market_data = self.market_frame(symbol)
        except ValueError:
            return False
        if market_data.empty:
            return False

        bar_ts = self.get_current_bar_timestamp(date, symbol)
        if bar_ts is None:
            return False
        try:
            current_idx = label_position(market_data.index, bar_ts)
        except KeyError:
            get_logger().warning("⚠️  Date not found in data")
            return False

        # Check if we have enough data to analyze (need at least 30 days for SMA 30)
        if current_idx < 30:
            get_logger().warning("⚠️  Not enough data to analyze")
            return False

        # Check if pre-calculated velocity data is available
        if "Velocity_Changes" not in market_data.columns:
            get_logger().warning("⚠️  No velocity changes data available")
            return False

        # Check if current velocity increased (positive velocity change)
        if current_idx < 1 or market_data["Velocity_Changes"].iloc[current_idx] <= 0:
            get_logger().info(f"Velocity changes: {market_data['Velocity_Changes'].iloc[current_idx]}")
            return False

        # This is a velocity signal - now check if it leads to a successful trend
        signal_index = current_idx

        # Check if this leads to an upward trend of at least 3 days, max 60 days
        success, duration, trend_return = self._check_trend_success(
            market_data, signal_index, "up", min_duration=3, max_duration=60
        )

        get_logger().info(f"Trend success: {success}, duration: {duration}, trend_return: {trend_return}")

        return success

    def _check_trend_success(
        self, data: pd.DataFrame, signal_index: int, trend_type: str, min_duration: int = 3, max_duration: int = 60
    ) -> tuple[bool, int, float]:
        """
        Check if a trend signal is part of a successful upward trend by looking backward.
        This method always uses backward analysis for consistency between backtesting and live trading.

        Note: This strategy only considers upward trends for momentum trading.

        Args:
            data: DataFrame with price data
            signal_index: Index of the signal
            trend_type: Only 'up' is supported (downward trends are ignored)
            min_duration: Minimum trend duration in days
            max_duration: Maximum trend duration in days

        Returns:
            Tuple of (success, duration, return)
        """
        # Always use backward trend analysis for consistency
        return self._check_backward_trend_success(data, signal_index, trend_type, min_duration, max_duration)

    def _check_backward_trend_success(
        self, data: pd.DataFrame, signal_index: int, trend_type: str, min_duration: int = 3, max_duration: int = 60
    ) -> tuple[bool, int, float]:
        """
        Check for successful upward trend by looking backward from the current date.
        This is used when we're on the current date and can't look forward.

        Note: This strategy only considers upward trends for momentum trading.

        Args:
            data: DataFrame with price data
            signal_index: Index of the signal (current date)
            trend_type: Only 'up' is supported (downward trends are ignored)
            min_duration: Minimum trend duration in days
            max_duration: Maximum trend duration in days

        Returns:
            Tuple of (success, duration, return)
        """
        current_price = data["Close"].iloc[signal_index]

        # Look backward for trend continuation
        for duration in range(min_duration, min(max_duration + 1, signal_index + 1)):
            start_index = signal_index - duration
            if start_index < 0:
                break

            start_price = data["Close"].iloc[start_index]
            total_return = (current_price - start_price) / start_price

            if trend_type == "up":
                get_logger().info(f"🔍 Backward up trend detected for {duration} days, total return: {total_return}")
                if total_return > 0:
                    # Check if this is a sustained upward trend
                    # Look for any significant reversal within the trend period
                    trend_sustained = True
                    for j in range(start_index + 1, signal_index):
                        current_price_in_trend = data["Close"].iloc[j]
                        current_return = (current_price_in_trend - start_price) / start_price
                        if current_return < -0.02:  # 2% reversal threshold
                            trend_sustained = False
                            break

                    if trend_sustained:
                        return True, duration, total_return
            # Note: This strategy only considers upward trends for momentum trading
            # Downward trends are not used as they don't align with the strategy's purpose

        return False, 0, 0.0

    def _create_put_credit_spread(
        self, date: datetime, current_price: float, expiration: str, symbol: str
    ) -> Optional[Position]:
        """
        Create a ATM/+10 put credit spread.

        Args:
            date: Current date
            current_price: Current underlying price
            expiration: Target expiration date string (YYYY-MM-DD)
            symbol: Underlying symbol

        Returns:
            Position: Position for evaluation, or None if creation fails
        """
        try:
            contracts = self._contracts_for_put_spread(date, current_price, expiration, symbol)
            if contracts is None:
                return None
            selected = self._select_put_spread_legs(contracts, current_price, expiration)
            if selected is None:
                return None
            atm_put, otm_put, atm_strike = selected
            return self._put_spread_position(date, expiration, atm_put, otm_put, atm_strike, symbol)
        except Exception as error:
            get_logger().warning(f"⚠️  Error creating put credit spread: {error}")
            return None

    def _contracts_for_put_spread(
        self, date: datetime, current_price: float, expiration: str, symbol: str
    ) -> Optional[list[OptionContractDTO]]:
        """Contracts for one expiration, limited to strikes near the underlying."""
        expiration_range = ExpirationRangeDTO(min_days=5, max_days=10)
        strike_range = StrikeRangeDTO(
            min_strike=StrikePrice(Decimal(str(current_price - 7))),
            max_strike=StrikePrice(Decimal(str(current_price + 1))),
        )
        contracts = self.get_contract_list_for_date(
            symbol, date, strike_range=strike_range, expiration_range=expiration_range
        )
        if not contracts:
            get_logger().warning("⚠️  No contracts found for the date")
            return None

        # Both legs must share this expiration so the result is a vertical spread.
        contracts_for_expiration = [contract for contract in contracts if str(contract.expiration_date) == expiration]
        if not contracts_for_expiration:
            get_logger().warning(f"⚠️  No contracts found for target expiration {expiration}")
            return None

        get_logger().info(f"✅ Found {len(contracts_for_expiration)} contracts for expiration {expiration}")
        return contracts_for_expiration

    def _select_put_spread_legs(
        self,
        contracts_for_expiration: Sequence[OptionContractDTO],
        current_price: float,
        expiration: str,
    ) -> Optional[tuple[OptionContractDTO, OptionContractDTO, float]]:
        """ATM short put and the put nearest six points below it, from the same expiration."""
        atm_strike = round(current_price)
        atm_put = OptionsRetrieverHelper.find_atm_contracts(contracts_for_expiration, current_price)[1]
        if not atm_put:
            get_logger().warning(f"⚠️  No ATM put found for expiration {expiration}")
            return None

        get_logger().info(
            f"Found ATM put: {atm_put.ticker} @ ${atm_put.strike_price.value} exp {atm_put.expiration_date}"
        )

        otm_strike = atm_strike - 6
        # Compare .value so tests stay isolated when enum identity differs across imports.
        puts_for_expiration = [
            contract for contract in contracts_for_expiration if contract.contract_type.value == OptionType.PUT.value
        ]
        if not puts_for_expiration:
            get_logger().warning(f"⚠️  No put contracts found for expiration {expiration}")
            return None

        otm_put = min(puts_for_expiration, key=lambda put: abs(float(put.strike_price.value) - otm_strike))
        min_difference = abs(float(otm_put.strike_price.value) - otm_strike)
        get_logger().info(
            f"Found OTM put: {otm_put.ticker} @ ${otm_put.strike_price.value} "
            f"exp {otm_put.expiration_date} (strike difference: {min_difference})"
        )

        if str(atm_put.expiration_date) != str(otm_put.expiration_date):
            get_logger().error(
                f"❌ ERROR: Expiration mismatch! ATM: {atm_put.expiration_date}, OTM: {otm_put.expiration_date}"
            )
            get_logger().error("❌ This would create a diagonal spread, not a vertical spread. Rejecting position.")
            return None

        get_logger().info(f"✅ Verified: Both legs have same expiration {expiration} (vertical spread)")
        return atm_put, otm_put, float(atm_strike)

    def _put_spread_position(
        self,
        date: datetime,
        expiration: str,
        atm_put: OptionContractDTO,
        otm_put: OptionContractDTO,
        atm_strike: float,
        symbol: str,
    ) -> Optional[Position]:
        """Build the credit-spread position from the two legs. Quantity is set by the engine."""
        atm_bar = self.get_current_option_bar(symbol, atm_put.ticker, date)
        otm_bar = self.get_current_option_bar(symbol, otm_put.ticker, date)
        if not atm_bar or not otm_bar:
            get_logger().warning("⚠️  No bar data available for credit calculation")
            return None

        net_credit = float(atm_bar.close_price) - float(otm_bar.close_price)
        if net_credit <= 0:
            get_logger().warning(f"⚠️  Negative credit: {net_credit:.2f}")
            return None

        atm_option = Option.from_contract_and_bar(atm_put, atm_bar)
        otm_option = Option.from_contract_and_bar(otm_put, otm_bar)
        return create_position(
            symbol=symbol,
            expiration_date=datetime.strptime(expiration, DateFormats.EXPIRATION),
            strategy_type=StrategyType.PUT_CREDIT_SPREAD,
            strike_price=atm_strike,
            entry_date=date,
            entry_price=net_credit,
            legs=[atm_option, otm_option],
        )

    def _get_risk_free_rate(self, date: datetime) -> float:
        """
        Get the risk-free rate for a specific date using the treasury data.

        Args:
            date: The date to get the risk-free rate for

        Returns:
            float: Risk-free rate (default 0.0 if not available)
        """
        if self.treasury_data is None:
            get_logger().warning("⚠️  No treasury data available, using default 2.0%")
            return 2.0

        return float(self.treasury_data.get_risk_free_rate(date))

    # ==== Helper methods (opening) ====
    def _try_open_position(self, date: datetime, add_position: Callable[["Position"], None], symbol: str):
        # Standard expiration target ~1 week
        if not self._has_buy_signal(date, symbol):
            return
        get_logger().info(f"📈 Buy signal detected for {date.strftime(DateFormats.ISO_DATE)}")

        current_price = self.get_current_underlying_price(date, symbol)
        if current_price is None:
            get_logger().warning("Failed to get current price.")
            return

        # Select expiration (target ~1 week)
        expiration_str = self._select_week_expiration(date, symbol)
        if not expiration_str:
            get_logger().warning("⚠️  Failed to select expiration")
            return

        position = self._create_put_credit_spread(date, current_price, expiration_str, symbol)
        if position is None:
            get_logger().warning("⚠️  Failed to create put credit spread for selected expiration")
            return

        get_logger().info(f"Current underlying price: {current_price}")

        # Track position entry for plotting
        self._position_entries.append(date)

        add_position(position)

    def _get_option_chain(self, date: datetime, current_price: float, symbol: str) -> Optional[OptionsChainDTO]:
        date_key = date.strftime(DateFormats.ISO_DATE)

        expiration_range = ExpirationRangeDTO(min_days=5, max_days=10)

        strike_range = StrikeRangeDTO(
            min_strike=StrikePrice(Decimal(str(current_price - 7))),  # current_price - 7 (width of 6 + buffer)
            max_strike=StrikePrice(Decimal(str(current_price + 1))),  # current_price + 1
        )

        live_chain = self.get_options_chain(
            symbol, date, current_price, strike_range=strike_range, expiration_range=expiration_range
        )
        if live_chain and (live_chain.get_calls() or live_chain.get_puts()):
            get_logger().info(
                f"✅ Successfully fetched live option chain with {len(live_chain.get_calls())} calls and {len(live_chain.get_puts())} puts"
            )
            self.options_data[date_key] = live_chain
            return live_chain

        get_logger().warning("⚠️  Live option chain fetch returned empty data")
        return None

    def _select_week_expiration(self, date: datetime, symbol: str) -> Optional[str]:
        """
        Select the best expiration date for the strategy using options_retriever.
        Prefer expirations 5-10 days out, else nearest > 0 days, target 7 days.
        """
        get_logger().info(f"🔍 _select_week_expiration called for {date.strftime(DateFormats.ISO_DATE)}")
        target_days = 7

        def days_out(exp_str: str) -> int:
            try:
                exp_dt = datetime.strptime(exp_str, DateFormats.EXPIRATION)
                return (exp_dt - date).days
            except Exception:
                return -9999

        # Try to get expirations from options_retriever
        try:
            get_logger().info("🔍 Fetching expirations from options_retriever for 5-10 day window...")

            # Use new_options_handler to get available expirations
            expiration_range = ExpirationRangeDTO(min_days=5, max_days=10)

            # Get contracts for the date (already filtered by expiration range)
            contracts = self.get_contract_list_for_date(symbol, date, expiration_range=expiration_range)

            if not contracts:
                get_logger().warning("⚠️  No contracts found for the date")
                return None

            # Extract unique expiration dates from contracts
            expirations = set(str(contract.expiration_date) for contract in contracts)
            get_logger().info(f"🔍 Found {len(expirations)} expirations from contracts")

            if not expirations:
                get_logger().warning("⚠️  No expirations found in option chain")
                return None

            # Calculate days out for each expiration and select closest to target (7 days)
            valid_expirations = [(e, days_out(e)) for e in expirations]
            valid_expirations = [(e, d) for e, d in valid_expirations if d > 0]

            if not valid_expirations:
                get_logger().warning("⚠️  No future expirations available")
                return None

            # Select the expiration closest to target (7 days)
            best_expiration = min(valid_expirations, key=lambda x: abs(x[1] - target_days))[0]
            days_to_exp = days_out(best_expiration)
            get_logger().info(f"✅ Selected expiration: {best_expiration} ({days_to_exp} days out)")

            return best_expiration

        except Exception as e:
            get_logger().error(f"❌ Error fetching expirations from options_retriever: {str(e)}")
            return None

    # ==== Helper methods (closing) ====
    def _try_close_positions(
        self,
        date: datetime,
        positions: tuple["Position", ...],
        remove_position: Strategy.RemovePositionCallback,
        symbol: str,
    ):
        current_underlying_price = self.get_current_underlying_price(date, symbol)
        get_logger().info(f"Current underlying price: {current_underlying_price}")
        get_logger().info(f"🤖 Strategy evaluating {len(positions)} open position(s) for potential closure...")

        for position in positions:
            days_held = position.get_days_held(date) if hasattr(position, "get_days_held") else 0
            days_to_exp = position.get_days_to_expiration(date) if hasattr(position, "get_days_to_expiration") else 0
            get_logger().info(f"🔍 Position {str(position)} - Days held: {days_held}, Days to exp: {days_to_exp}")

            # Compute exit price for stop/holding decisions
            exit_price = self.compute_exit_price(position, date)
            if exit_price is None:
                get_logger().warning(f"⚠️  compute_exit_price not configured; skipping closure checks for {position}")
                continue
            if exit_price is not None:
                exit_price = self._sanitize_exit_price(exit_price)
                get_logger().info(f"💰 Calculated exit price for {str(position)}: {exit_price}")

            # Holding period
            if self._should_close_due_to_holding(position, date, self.holding_period):
                if exit_price is not None:
                    get_logger().info(
                        f"Holding period met for {str(position)} at exit {exit_price} (held {days_held} days, target: {self.holding_period})"
                    )
                    current_volumes = self.get_current_volumes_for_position(position, date)
                    remove_position(
                        date,
                        position,
                        exit_price,
                        underlying_price=current_underlying_price,
                        current_volumes=current_volumes,
                    )
                else:
                    get_logger().warning(
                        f"⚠️  No exit price available for {str(position)} on {date}. Skipping holding-period close."
                    )
            else:
                # Position not closed - show why
                get_logger().info(
                    f"📋 Position {str(position)} remains open - Days held: {days_held}/{self.holding_period}, Days to exp: {days_to_exp}"
                )
                get_logger().info(
                    f"🤖 Strategy decision: Position does not meet closing criteria (holding period: {self.holding_period} days, stop loss, or expiration)"
                )

        # Summary of strategy decisions
        get_logger().info(
            f"✅ Strategy evaluation complete - no positions closed on {date.strftime(DateFormats.ISO_DATE)}"
        )

    def _sanitize_exit_price(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return round(max(value, 0), 2)

    def _should_close_due_to_holding(self, position: Position, date: datetime, holding_period: int) -> bool:
        try:
            return position.get_days_held(date) >= holding_period
        except Exception:
            return False
