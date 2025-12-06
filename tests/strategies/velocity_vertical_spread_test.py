"""
Test to verify that velocity strategy creates vertical spreads (not diagonal spreads).
A vertical spread must have both legs with the SAME expiration date.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch

from src.strategies.velocity_signal_momentum_strategy import VelocitySignalMomentumStrategy
from src.common.models import OptionType
from src.common.options_dtos import (
    OptionContractDTO, 
    OptionBarDTO, 
    StrikePrice,
    ExpirationRangeDTO,
    StrikeRangeDTO
)


class TestVelocityVerticalSpread:
    """Test that velocity strategy only creates vertical spreads (same expiration for both legs)."""
    
    def test_put_credit_spread_has_same_expiration_for_both_legs(self):
        """
        Test that _create_put_credit_spread returns a position where both options
        have the same expiration date (vertical spread, not diagonal).
        """
        # Setup
        strategy = VelocitySignalMomentumStrategy(
            options_handler=Mock(),
            start_date_offset=60
        )
        
        current_date = datetime(2025, 10, 30, 14, 0, 0)
        current_price = 585.0
        target_expiration = "2025-11-06"  # Target expiration (7 days out)
        
        # Create mock contracts with MULTIPLE expirations to simulate real scenario
        # where different expirations are available
        exp_date_1 = datetime(2025, 11, 4)  # 5 days out
        exp_date_2 = datetime(2025, 11, 6)  # 7 days out (target)
        exp_date_3 = datetime(2025, 11, 7)  # 8 days out
        
        from src.common.options_dtos import ExpirationDate
        from datetime import date as date_type
        
        mock_contracts = [
            # Contracts expiring Nov 4 (5 days out)
            OptionContractDTO(
                ticker="O:SPY251104P585",
                underlying_ticker="SPY",
                strike_price=StrikePrice(Decimal("585")),
                expiration_date=ExpirationDate(date_type(2025, 11, 4)),
                contract_type=OptionType.PUT,
                exercise_style='american',
                shares_per_contract=100
            ),
            OptionContractDTO(
                ticker="O:SPY251104P579",
                underlying_ticker="SPY",
                strike_price=StrikePrice(Decimal("579")),
                expiration_date=ExpirationDate(date_type(2025, 11, 4)),
                contract_type=OptionType.PUT,
                exercise_style='american',
                shares_per_contract=100
            ),
            # Contracts expiring Nov 6 (7 days out - TARGET)
            OptionContractDTO(
                ticker="O:SPY251106P585",
                underlying_ticker="SPY",
                strike_price=StrikePrice(Decimal("585")),
                expiration_date=ExpirationDate(date_type(2025, 11, 6)),
                contract_type=OptionType.PUT,
                exercise_style='american',
                shares_per_contract=100
            ),
            OptionContractDTO(
                ticker="O:SPY251106P579",
                underlying_ticker="SPY",
                strike_price=StrikePrice(Decimal("579")),
                expiration_date=ExpirationDate(date_type(2025, 11, 6)),
                contract_type=OptionType.PUT,
                exercise_style='american',
                shares_per_contract=100
            ),
            # Contracts expiring Nov 7 (8 days out)
            OptionContractDTO(
                ticker="O:SPY251107P585",
                underlying_ticker="SPY",
                strike_price=StrikePrice(Decimal("585")),
                expiration_date=ExpirationDate(date_type(2025, 11, 7)),
                contract_type=OptionType.PUT,
                exercise_style='american',
                shares_per_contract=100
            ),
            OptionContractDTO(
                ticker="O:SPY251107P579",
                underlying_ticker="SPY",
                strike_price=StrikePrice(Decimal("579")),
                expiration_date=ExpirationDate(date_type(2025, 11, 7)),
                contract_type=OptionType.PUT,
                exercise_style='american',
                shares_per_contract=100
            ),
        ]
        
        # Create mock bar data for pricing
        mock_bar_585 = OptionBarDTO(
            ticker="O:SPY251106P585",
            timestamp=current_date,
            open_price=Decimal("3.50"),
            high_price=Decimal("3.60"),
            low_price=Decimal("3.40"),
            close_price=Decimal("3.55"),
            volume=1000,
            volume_weighted_avg_price=Decimal("3.55"),
            number_of_transactions=100
        )
        
        mock_bar_579 = OptionBarDTO(
            ticker="O:SPY251106P579",
            timestamp=current_date,
            open_price=Decimal("2.80"),
            high_price=Decimal("2.90"),
            low_price=Decimal("2.75"),
            close_price=Decimal("2.85"),
            volume=800,
            volume_weighted_avg_price=Decimal("2.85"),
            number_of_transactions=80
        )
        
        # Mock the new_options_handler methods
        strategy.new_options_handler.get_contract_list_for_date = Mock(return_value=mock_contracts)
        
        # Mock get_option_bar to return appropriate bars based on contract
        def mock_get_option_bar(option, date):
            if "585" in option.ticker:
                return mock_bar_585
            elif "579" in option.ticker:
                return mock_bar_579
            return None
        
        strategy.new_options_handler.get_option_bar = Mock(side_effect=mock_get_option_bar)
        
        # Set up data for the strategy (required for position creation)
        import pandas as pd
        data = pd.DataFrame({
            'Close': [580, 582, 585]
        }, index=pd.date_range('2025-10-28', periods=3))
        strategy.set_data(data, {})
        
        # Execute
        position = strategy._create_put_credit_spread(
            date=current_date,
            current_price=current_price,
            expiration=target_expiration
        )
        
        # Assert
        assert position is not None, "Position should be created"
        assert len(position.spread_options) == 2, "Position should have 2 legs"
        
        short_leg = position.spread_options[0]  # ATM put (sold)
        long_leg = position.spread_options[1]   # OTM put (bought)
        
        # Extract expiration dates from Option objects (they use 'expiration' attribute, not 'expiration_date')
        short_exp = short_leg.expiration if hasattr(short_leg, 'expiration') else None
        long_exp = long_leg.expiration if hasattr(long_leg, 'expiration') else None
        
        print(f"\n🔍 Debug Info:")
        print(f"   Short leg: {short_leg.ticker}, expiration: {short_exp}")
        print(f"   Long leg: {long_leg.ticker}, expiration: {long_exp}")
        print(f"   Target expiration: {target_expiration}")
        
        # CRITICAL: Both legs must have the same expiration (vertical spread)
        assert short_exp == long_exp, (
            f"Both legs must have the same expiration date for a vertical spread. "
            f"Got short_leg={short_exp}, long_leg={long_exp}. "
            f"This is a diagonal spread, not a vertical spread!"
        )
        
        # Additionally verify they match the target expiration
        # Note: Option.expiration is a string, not a datetime
        assert short_exp == target_expiration, f"Short leg expiration {short_exp} doesn't match target {target_expiration}"
        assert long_exp == target_expiration, f"Long leg expiration {long_exp} doesn't match target {target_expiration}"
        
        print(f"✅ Test passed: Both legs have the same expiration date ({short_exp})")
        print(f"✅ This is a proper vertical spread!")


    def test_rejects_position_if_no_matching_expiration_contracts(self):
        """
        Test that the strategy returns None if it cannot find contracts
        with the target expiration for both legs.
        """
        strategy = VelocitySignalMomentumStrategy(
            options_handler=Mock(),
            start_date_offset=60
        )
        
        current_date = datetime(2025, 10, 30, 14, 0, 0)
        current_price = 585.0
        target_expiration = "2025-11-06"
        
        # Create contracts that DON'T include the target expiration
        from src.common.options_dtos import ExpirationDate
        from datetime import date as date_type
        
        mock_contracts = [
            OptionContractDTO(
                ticker="O:SPY251108P585",
                underlying_ticker="SPY",
                strike_price=StrikePrice(Decimal("585")),
                expiration_date=ExpirationDate(date_type(2025, 11, 8)),
                contract_type=OptionType.PUT,
                exercise_style='american',
                shares_per_contract=100
            ),
        ]
        
        strategy.new_options_handler.get_contract_list_for_date = Mock(return_value=mock_contracts)
        
        # Execute
        position = strategy._create_put_credit_spread(
            date=current_date,
            current_price=current_price,
            expiration=target_expiration
        )
        
        # Assert - should return None if target expiration not available
        assert position is None, (
            "Position should be None if target expiration contracts are not available"
        )
        print("✅ Test passed: Strategy correctly rejects position when target expiration unavailable")


if __name__ == "__main__":
    # Run the tests
    test = TestVelocityVerticalSpread()
    
    print("="*80)
    print("TEST 1: Verify both legs have same expiration (vertical spread)")
    print("="*80)
    try:
        test.test_put_credit_spread_has_same_expiration_for_both_legs()
        print("\n✅ TEST 1 PASSED\n")
    except AssertionError as e:
        print(f"\n❌ TEST 1 FAILED: {e}\n")
    except Exception as e:
        print(f"\n❌ TEST 1 ERROR: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("="*80)
    print("TEST 2: Reject position if target expiration unavailable")
    print("="*80)
    try:
        test.test_rejects_position_if_no_matching_expiration_contracts()
        print("\n✅ TEST 2 PASSED\n")
    except AssertionError as e:
        print(f"\n❌ TEST 2 FAILED: {e}\n")
    except Exception as e:
        print(f"\n❌ TEST 2 ERROR: {e}\n")
        import traceback
        traceback.print_exc()

