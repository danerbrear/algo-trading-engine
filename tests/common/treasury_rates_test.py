import pytest
import pandas as pd
from datetime import datetime
from decimal import Decimal
from algo_trading_engine.common.models import TreasuryRates


class TestTreasuryRates:
    """Test cases for TreasuryRates Value Object"""

    def setup_method(self):
        """Set up test data"""
        # Create sample treasury rates data
        dates = pd.date_range('2021-01-04', '2021-01-08', freq='D')
        self.sample_data = pd.DataFrame({
            'IRX_1Y': [0.00068, 0.00078, 0.00078, 0.00080, 0.00080],
            'TNX_10Y': [0.00917, 0.00955, 0.01042, 0.01071, 0.01105]
        }, index=dates)

    def test_valid_creation(self):
        """Test creating TreasuryRates with valid data"""
        treasury_rates = TreasuryRates(self.sample_data)
        assert treasury_rates.rates_data is not None
        assert len(treasury_rates.rates_data) == 5

    def test_creation_with_empty_data(self):
        """Test creating TreasuryRates with empty data should raise ValueError"""
        empty_data = pd.DataFrame(columns=['IRX_1Y', 'TNX_10Y'])
        with pytest.raises(ValueError, match="Treasury rates data cannot be empty"):
            TreasuryRates(empty_data)

    def test_creation_with_none_data(self):
        """Test creating TreasuryRates with None data should raise ValueError"""
        with pytest.raises(ValueError, match="Treasury rates data cannot be empty"):
            TreasuryRates(None)

    def test_creation_with_missing_columns(self):
        """Test creating TreasuryRates with missing required columns should raise ValueError"""
        invalid_data = pd.DataFrame({
            'IRX_1Y': [0.00068, 0.00078]
        })
        with pytest.raises(ValueError, match="Missing required treasury rate columns"):
            TreasuryRates(invalid_data)

    def test_get_risk_free_rate_exact_date(self):
        """Test getting risk-free rate for exact date"""
        treasury_rates = TreasuryRates(self.sample_data)
        test_date = datetime(2021, 1, 4)
        rate = treasury_rates.get_risk_free_rate(test_date)
        assert rate == Decimal('0.00068')

    def test_get_risk_free_rate_closest_date(self):
        """Test getting risk-free rate for date not in data"""
        treasury_rates = TreasuryRates(self.sample_data)
        test_date = datetime(2021, 1, 3)  # Date not in data
        rate = treasury_rates.get_risk_free_rate(test_date)
        # Should return closest date (2021-01-04)
        assert rate == Decimal('0.00068')

    def test_get_risk_free_rate_fallback(self):
        """Test getting risk-free rate when no data available"""
        empty_data = pd.DataFrame(columns=['IRX_1Y', 'TNX_10Y'])
        # This will fail creation, so we need to handle it differently
        with pytest.raises(ValueError):
            TreasuryRates(empty_data)

    def test_get_10_year_rate_exact_date(self):
        """Test getting 10-year rate for exact date"""
        treasury_rates = TreasuryRates(self.sample_data)
        test_date = datetime(2021, 1, 4)
        rate = treasury_rates.get_10_year_rate(test_date)
        assert rate == Decimal('0.00917')

    def test_get_10_year_rate_closest_date(self):
        """Test getting 10-year rate for date not in data"""
        treasury_rates = TreasuryRates(self.sample_data)
        test_date = datetime(2021, 1, 3)  # Date not in data
        rate = treasury_rates.get_10_year_rate(test_date)
        # Should return closest date (2021-01-04)
        assert rate == Decimal('0.00917')

    def test_get_date_range(self):
        """Test getting date range of treasury rates data"""
        treasury_rates = TreasuryRates(self.sample_data)
        start_date, end_date = treasury_rates.get_date_range()
        assert start_date == pd.Timestamp('2021-01-04')
        assert end_date == pd.Timestamp('2021-01-08')

    def test_is_empty(self):
        """Test is_empty method"""
        treasury_rates = TreasuryRates(self.sample_data)
        assert not treasury_rates.is_empty()

    def test_immutability(self):
        """Test that TreasuryRates is immutable"""
        treasury_rates = TreasuryRates(self.sample_data)
        
        # Should not be able to modify the rates_data
        with pytest.raises(AttributeError):
            treasury_rates.rates_data = None

    def test_value_equality(self):
        """Test that two TreasuryRates with same data are equal"""
        treasury_rates1 = TreasuryRates(self.sample_data)
        treasury_rates2 = TreasuryRates(self.sample_data.copy())
        assert treasury_rates1 == treasury_rates2

    def test_hash_consistency(self):
        """Test that TreasuryRates can be used as dictionary key"""
        treasury_rates = TreasuryRates(self.sample_data)
        test_dict = {treasury_rates: "test_value"}
        assert test_dict[treasury_rates] == "test_value"
