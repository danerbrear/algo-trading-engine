"""
Helper classes for options data retrieval and processing.

This module contains static utility methods for common options operations
as specified in features/improved_data_fetching.md Phase 4.
"""

from typing import Callable, List, Tuple, Optional, Dict, Any
from decimal import Decimal
from datetime import date, datetime

from algo_trading_engine.dto import OptionContractDTO, StrikeRangeDTO, ExpirationRangeDTO, OptionBarDTO, OptionsChainDTO
from .models import OptionType


class OptionsRetrieverHelper:
    """
    Static helper methods for options data retrieval and processing.
    
    These methods provide common operations for filtering, finding, and
    calculating options data without requiring instance state.
    """
    
    @staticmethod
    def filter_contracts_by_strike(
        contracts: List[OptionContractDTO], 
        target_strike: float, 
        tolerance: float
    ) -> List[OptionContractDTO]:
        """
        Filter contracts within strike tolerance.
        
        Args:
            contracts: List of option contracts
            target_strike: Target strike price
            tolerance: Strike tolerance (absolute value)
            
        Returns:
            List of contracts within tolerance
        """
        target_decimal = Decimal(str(target_strike))
        tolerance_decimal = Decimal(str(tolerance))
        
        filtered = []
        for contract in contracts:
            # Convert strike price to Decimal for comparison
            strike_decimal = Decimal(str(contract.strike_price.value))
            strike_diff = abs(strike_decimal - target_decimal)
            if strike_diff <= tolerance_decimal:
                filtered.append(contract)
        
        return filtered
    
    @staticmethod
    def find_atm_contracts(
        contracts: List[OptionContractDTO], 
        current_price: float
    ) -> Tuple[Optional[OptionContractDTO], Optional[OptionContractDTO]]:
        """
        Find ATM call and put contracts.
        
        Args:
            contracts: List of option contracts
            current_price: Current underlying price
            
        Returns:
            Tuple of (call_contract, put_contract) or (None, None) if not found
        """
        current_decimal = Decimal(str(current_price))
        
        # Find call and put contracts with minimum distance from current price
        # Use .value comparison to handle test isolation issues where enum identity might differ
        call_contracts = [c for c in contracts if c.contract_type.value == OptionType.CALL.value]
        put_contracts = [c for c in contracts if c.contract_type.value == OptionType.PUT.value]
        
        call_contract = (
            min(call_contracts, key=lambda c: abs(Decimal(str(c.strike_price.value)) - current_decimal))
            if call_contracts else None
        )
        
        put_contract = (
            min(put_contracts, key=lambda c: abs(Decimal(str(c.strike_price.value)) - current_decimal))
            if put_contracts else None
        )
        
        return call_contract, put_contract
    
    
    @staticmethod
    def find_contracts_by_expiration(
        contracts: List[OptionContractDTO], 
        target_expiration: str
    ) -> List[OptionContractDTO]:
        """
        Find contracts with specific expiration date.
        
        Args:
            contracts: List of option contracts
            target_expiration: Target expiration date (YYYY-MM-DD)
            
        Returns:
            List of contracts with matching expiration
        """
        filtered = []
        for contract in contracts:
            if str(contract.expiration_date) == target_expiration:
                filtered.append(contract)
        
        return filtered
    
    @staticmethod
    def find_contracts_by_type(
        contracts: List[OptionContractDTO], 
        option_type: OptionType
    ) -> List[OptionContractDTO]:
        """
        Find contracts that match the given option type.
        
        Args:
            contracts: List of option contracts
            option_type: Target option type (CALL or PUT)
            
        Returns:
            List[OptionContractDTO]: Filtered contracts matching option type
        """
        matching_contracts = []
        for contract in contracts:
            if contract.contract_type == option_type:
                matching_contracts.append(contract)
        return matching_contracts
    
    @staticmethod
    def find_itm_contracts(
        contracts: List[OptionContractDTO], 
        current_price: float
    ) -> List[OptionContractDTO]:
        """
        Find in-the-money contracts.
        
        Args:
            contracts: List of option contracts
            current_price: Current underlying price
            
        Returns:
            List of ITM contracts
        """
        current_decimal = Decimal(str(current_price))
        itm_contracts = []
        
        for contract in contracts:
            # Convert strike price to Decimal for comparison
            strike_decimal = Decimal(str(contract.strike_price.value))
            if contract.contract_type == OptionType.CALL:
                is_itm = strike_decimal < current_decimal
            else:  # PUT
                is_itm = strike_decimal > current_decimal
            
            if is_itm:
                itm_contracts.append(contract)
        
        return itm_contracts
    
    @staticmethod
    def find_otm_contracts(
        contracts: List[OptionContractDTO], 
        current_price: float
    ) -> List[OptionContractDTO]:
        """
        Find out-of-the-money contracts.
        
        Args:
            contracts: List of option contracts
            current_price: Current underlying price
            
        Returns:
            List of OTM contracts
        """
        current_decimal = Decimal(str(current_price))
        tolerance = Decimal('0.01')  # 1 cent tolerance
        otm_contracts = []
        
        for contract in contracts:
            # Convert strike price to Decimal for comparison
            strike_decimal = Decimal(str(contract.strike_price.value))
            
            # Check if ATM first
            is_atm = abs(strike_decimal - current_decimal) <= tolerance
            
            if not is_atm:
                # Check if ITM
                if contract.contract_type == OptionType.CALL:
                    is_itm = strike_decimal < current_decimal
                else:  # PUT
                    is_itm = strike_decimal > current_decimal
                
                # OTM if not ATM and not ITM
                if not is_itm:
                    otm_contracts.append(contract)
        
        return otm_contracts
    
    @staticmethod
    def sort_contracts_by_strike(
        contracts: List[OptionContractDTO], 
        ascending: bool = True
    ) -> List[OptionContractDTO]:
        """
        Sort contracts by strike price.
        
        Args:
            contracts: List of option contracts
            ascending: Sort order (True for ascending, False for descending)
            
        Returns:
            Sorted list of contracts
        """
        return sorted(contracts, key=lambda c: c.strike_price.value, reverse=not ascending)
    
    @staticmethod
    def sort_contracts_by_expiration(
        contracts: List[OptionContractDTO], 
        ascending: bool = True
    ) -> List[OptionContractDTO]:
        """
        Sort contracts by expiration date.
        
        Args:
            contracts: List of option contracts
            ascending: Sort order (True for ascending, False for descending)
            
        Returns:
            Sorted list of contracts
        """
        return sorted(contracts, key=lambda c: c.expiration_date.date, reverse=not ascending)
    
    @staticmethod
    def find_closest_strike_contract(
        contracts: List[OptionContractDTO], 
        target_strike: float
    ) -> Optional[OptionContractDTO]:
        """
        Find contract with closest strike to target.
        
        Args:
            contracts: List of option contracts
            target_strike: Target strike price
            
        Returns:
            Contract with closest strike, or None if empty list
        """
        if not contracts:
            return None
        
        target_decimal = Decimal(str(target_strike))
        
        closest_contract = min(
            contracts, 
            key=lambda c: abs(Decimal(str(c.strike_price.value)) - target_decimal)
        )
        
        return closest_contract
    
    @staticmethod
    def find_contracts_in_strike_range(
        contracts: List[OptionContractDTO], 
        min_strike: float, 
        max_strike: float
    ) -> List[OptionContractDTO]:
        """
        Find contracts within strike range.
        
        Args:
            contracts: List of option contracts
            min_strike: Minimum strike price
            max_strike: Maximum strike price
            
        Returns:
            List of contracts within range
        """
        min_decimal = Decimal(str(min_strike))
        max_decimal = Decimal(str(max_strike))
        
        filtered = []
        for contract in contracts:
            if min_decimal <= contract.strike_price.value <= max_decimal:
                filtered.append(contract)
        
        return filtered
    
    @staticmethod
    def find_contracts_by_days_to_expiration(
        contracts: List[OptionContractDTO], 
        min_days: int, 
        max_days: int
    ) -> List[OptionContractDTO]:
        """
        Find contracts within days to expiration range.
        
        Args:
            contracts: List of option contracts
            min_days: Minimum days to expiration
            max_days: Maximum days to expiration
            
        Returns:
            List of contracts within range
        """
        filtered = []
        for contract in contracts:
            days_to_exp = contract.days_to_expiration()
            if min_days <= days_to_exp <= max_days:
                filtered.append(contract)
        
        return filtered
    
    @staticmethod
    def group_contracts_by_expiration(
        contracts: List[OptionContractDTO]
    ) -> dict:
        """
        Group contracts by expiration date.
        
        Args:
            contracts: List of option contracts
            
        Returns:
            Dict mapping expiration date to list of contracts
        """
        grouped = {}
        for contract in contracts:
            exp_date = str(contract.expiration_date)
            if exp_date not in grouped:
                grouped[exp_date] = []
            grouped[exp_date].append(contract)
        
        return grouped
    
    @staticmethod
    def group_contracts_by_strike(
        contracts: List[OptionContractDTO]
    ) -> dict:
        """
        Group contracts by strike price.
        
        Args:
            contracts: List of option contracts
            
        Returns:
            Dict mapping strike price to list of contracts
        """
        grouped = {}
        for contract in contracts:
            strike = float(contract.strike_price.value)
            if strike not in grouped:
                grouped[strike] = []
            grouped[strike].append(contract)
        
        return grouped
    
    @staticmethod
    def calculate_contract_statistics(
        contracts: List[OptionContractDTO]
    ) -> dict:
        """
        Calculate statistics for a list of contracts.
        
        Args:
            contracts: List of option contracts
            
        Returns:
            Dict with contract statistics
        """
        if not contracts:
            return {
                'total_contracts': 0,
                'calls': 0,
                'puts': 0,
                'unique_strikes': 0,
                'unique_expirations': 0,
                'strike_range': None,
                'expiration_range': None
            }
        
        calls = sum(1 for c in contracts if c.contract_type == OptionType.CALL)
        puts = sum(1 for c in contracts if c.contract_type == OptionType.PUT)
        
        strikes = [float(c.strike_price.value) for c in contracts]
        expirations = [c.expiration_date.date for c in contracts]
        
        return {
            'total_contracts': len(contracts),
            'calls': calls,
            'puts': puts,
            'unique_strikes': len(set(strikes)),
            'unique_expirations': len(set(expirations)),
            'strike_range': (min(strikes), max(strikes)) if strikes else None,
            'expiration_range': (min(expirations), max(expirations)) if expirations else None
        }
    
    @staticmethod
    def validate_contract_data(contract: OptionContractDTO) -> List[str]:
        """
        Validate contract data and return list of issues.
        
        Args:
            contract: Contract to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Check ticker format
        if not contract.ticker.startswith('O:'):
            issues.append(f"Invalid ticker format: {contract.ticker}")
        
        # Check strike price
        if contract.strike_price.value <= 0:
            issues.append(f"Invalid strike price: {contract.strike_price.value}")
        
        # Check expiration date
        if contract.expiration_date.date < contract.expiration_date.date.today():
            issues.append(f"Expiration date in past: {contract.expiration_date}")
        
        # Check shares per contract
        if contract.shares_per_contract <= 0:
            issues.append(f"Invalid shares per contract: {contract.shares_per_contract}")
        
        return issues
    
    # Strategy-specific helper methods
    
    @staticmethod
    def find_credit_spread_legs(
        contracts: List[OptionContractDTO], 
        current_price: float, 
        expiration_date: str,
        option_type: OptionType,
        spread_width: int = 5
    ) -> Tuple[Optional[OptionContractDTO], Optional[OptionContractDTO]]:
        """
        Find short and long legs for a credit spread strategy.
        
        Args:
            contracts: List of option contracts
            current_price: Current underlying price
            expiration_date: Target expiration date
            option_type: Type of spread (CALL or PUT)
            spread_width: Width of the spread in dollars
            
        Returns:
            Tuple of (short_leg, long_leg) or (None, None) if not found
        """
        # Filter contracts by expiration and type
        exp_contracts = OptionsRetrieverHelper.find_contracts_by_expiration(contracts, expiration_date)
        type_contracts = OptionsRetrieverHelper.find_contracts_by_type(exp_contracts, option_type)
        
        if len(type_contracts) < 2:
            return None, None
        
        # Sort by strike price
        sorted_contracts = OptionsRetrieverHelper.sort_contracts_by_strike(type_contracts)
        
        current_decimal = Decimal(str(current_price))
        
        if option_type == OptionType.CALL:
            # For call credit spread: short ATM, long OTM
            # Find ATM call (short leg)
            short_leg = None
            for contract in sorted_contracts:
                if contract.strike_price.value <= current_decimal:
                    short_leg = contract
                else:
                    break
            
            if short_leg:
                # Find long leg (spread_width above short leg)
                target_long_strike = Decimal(str(short_leg.strike_price.value)) + Decimal(str(spread_width))
                long_leg = OptionsRetrieverHelper.find_closest_strike_contract(
                    sorted_contracts, float(target_long_strike)
                )
                # Ensure long leg is different from short leg and has higher strike
                if long_leg and long_leg != short_leg and long_leg.strike_price.value > short_leg.strike_price.value:
                    return short_leg, long_leg
                else:
                    # If no suitable long leg found, try to find any contract with higher strike
                    for contract in sorted_contracts:
                        if contract != short_leg and contract.strike_price.value > short_leg.strike_price.value:
                            return short_leg, contract
        
        else:  # PUT
            # For put credit spread: short ATM, long OTM
            # Find ATM put (short leg)
            short_leg = None
            for contract in reversed(sorted_contracts):
                if contract.strike_price.value >= current_decimal:
                    short_leg = contract
                else:
                    break
            
            if short_leg:
                # Find long leg (spread_width below short leg)
                target_long_strike = Decimal(str(short_leg.strike_price.value)) - Decimal(str(spread_width))
                long_leg = OptionsRetrieverHelper.find_closest_strike_contract(
                    sorted_contracts, float(target_long_strike)
                )
                # Ensure long leg is different from short leg and has lower strike
                if long_leg and long_leg != short_leg and long_leg.strike_price.value < short_leg.strike_price.value:
                    return short_leg, long_leg
                else:
                    # If no suitable long leg found, try to find any contract with lower strike
                    for contract in sorted_contracts:
                        if contract != short_leg and contract.strike_price.value < short_leg.strike_price.value:
                            return short_leg, contract
        
        return None, None
    
    @staticmethod
    def calculate_credit_spread_premium(
        short_leg: OptionContractDTO, 
        long_leg: OptionContractDTO,
        short_premium: float,
        long_premium: float
    ) -> float:
        """
        Calculate net credit received for a credit spread.
        
        Args:
            short_leg: Short leg contract
            long_leg: Long leg contract
            short_premium: Premium received for short leg
            long_premium: Premium paid for long leg
            
        Returns:
            Net credit received
        """
        return short_premium - long_premium
    
    
    @staticmethod
    def find_optimal_expiration(
        contracts: List[OptionContractDTO], 
        min_days: int = 20, 
        max_days: int = 40,
        current_date: Optional[date] = None
    ) -> Optional[str]:
        """
        Find optimal expiration date for trading strategies.
        
        Args:
            contracts: List of option contracts
            min_days: Minimum days to expiration
            max_days: Maximum days to expiration
            current_date: Current date (defaults to today)
            
        Returns:
            Optimal expiration date string or None if not found
        """
        if current_date is None:
            current_date = date.today()
        
        # Group contracts by expiration
        grouped = OptionsRetrieverHelper.group_contracts_by_expiration(contracts)
        
        # Find expiration with most contracts in the desired range
        best_expiration = None
        max_contracts = 0
        
        for exp_date_str, exp_contracts in grouped.items():
            try:
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                days_to_exp = (exp_date - current_date).days
                
                if min_days <= days_to_exp <= max_days:
                    if len(exp_contracts) > max_contracts:
                        max_contracts = len(exp_contracts)
                        best_expiration = exp_date_str
            except ValueError:
                continue
        
        return best_expiration
    
    @staticmethod
    def calculate_implied_volatility_rank(
        contracts: List[OptionContractDTO], 
        current_price: float,
        lookback_days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate implied volatility rank for contracts.
        
        Note: This is a simplified implementation. In practice, you would
        need historical IV data to calculate proper IV rank.
        
        Args:
            contracts: List of option contracts
            current_price: Current underlying price
            lookback_days: Days to look back for IV calculation
            
        Returns:
            Dict mapping contract ticker to IV rank (0-100)
        """
        # This is a placeholder implementation
        # In practice, you would need historical IV data
        iv_ranks = {}
        
        for contract in contracts:
            # Simplified IV rank calculation
            # In practice, you would compare current IV to historical IV range
            days_to_exp = contract.days_to_expiration()
            
            # Placeholder logic: shorter DTE = higher IV rank
            if days_to_exp <= 7:
                iv_rank = 80.0
            elif days_to_exp <= 14:
                iv_rank = 60.0
            elif days_to_exp <= 30:
                iv_rank = 40.0
            else:
                iv_rank = 20.0
            
            iv_ranks[contract.ticker] = iv_rank
        
        return iv_ranks
    
    @staticmethod
    def find_high_volume_contracts(
        contracts: List[OptionContractDTO], 
        bars: Dict[str, OptionBarDTO],
        min_volume: int = 100
    ) -> List[OptionContractDTO]:
        """
        Find contracts with high volume based on bar data.
        
        Args:
            contracts: List of option contracts
            bars: Dict mapping ticker to bar data
            min_volume: Minimum volume threshold
            
        Returns:
            List of high volume contracts
        """
        high_volume_contracts = []
        
        for contract in contracts:
            bar = bars.get(contract.ticker)
            if bar and bar.volume >= min_volume:
                high_volume_contracts.append(contract)
        
        return high_volume_contracts
    
    @staticmethod
    def calculate_delta_exposure(
        contracts: List[OptionContractDTO], 
        bars: Dict[str, OptionBarDTO],
        quantity: int = 1
    ) -> float:
        """
        Calculate total delta exposure for a list of contracts.
        
        Note: This is a simplified implementation. In practice, you would
        need actual delta values from the options data.
        
        Args:
            contracts: List of option contracts
            bars: Dict mapping ticker to bar data
            quantity: Number of contracts
            
        Returns:
            Total delta exposure
        """
        total_delta = 0.0
        
        for contract in contracts:
            # Simplified delta calculation based on moneyness
            # In practice, you would use actual delta values
            current_price = 600.0  # This should be passed as parameter
            strike = float(contract.strike_price.value)
            
            if contract.contract_type == OptionType.CALL:
                # Simplified call delta: closer to ATM = higher delta
                moneyness = strike / current_price
                if moneyness < 0.95:
                    delta = 0.8  # Deep ITM
                elif moneyness < 1.05:
                    delta = 0.5  # ATM
                else:
                    delta = 0.2  # OTM
            else:  # PUT
                # Simplified put delta: closer to ATM = higher negative delta
                moneyness = strike / current_price
                if moneyness > 1.05:
                    delta = -0.8  # Deep ITM
                elif moneyness > 0.95:
                    delta = -0.5  # ATM
                else:
                    delta = -0.2  # OTM
            
            total_delta += delta * quantity
        
        return total_delta
    
    @staticmethod
    def find_iron_condor_legs(
        contracts: List[OptionContractDTO], 
        current_price: float, 
        expiration_date: str,
        spread_width: int = 5
    ) -> Tuple[Optional[OptionContractDTO], Optional[OptionContractDTO], 
               Optional[OptionContractDTO], Optional[OptionContractDTO]]:
        """
        Find all four legs for an iron condor strategy.
        
        Args:
            contracts: List of option contracts
            current_price: Current underlying price
            expiration_date: Target expiration date
            spread_width: Width of each spread
            
        Returns:
            Tuple of (put_long, put_short, call_short, call_long) or (None, None, None, None)
        """
        # Filter contracts by expiration
        exp_contracts = OptionsRetrieverHelper.find_contracts_by_expiration(contracts, expiration_date)
        
        if len(exp_contracts) < 4:
            return None, None, None, None
        
        # Find put credit spread legs
        put_short, put_long = OptionsRetrieverHelper.find_credit_spread_legs(
            exp_contracts, current_price, expiration_date, OptionType.PUT, spread_width
        )
        
        # Find call credit spread legs
        call_short, call_long = OptionsRetrieverHelper.find_credit_spread_legs(
            exp_contracts, current_price, expiration_date, OptionType.CALL, spread_width
        )
        
        if all([put_short, put_long, call_short, call_long]):
            return put_long, put_short, call_short, call_long
        
        return None, None, None, None
    
    @staticmethod
    def calculate_breakeven_points(
        short_leg: OptionContractDTO, 
        long_leg: OptionContractDTO,
        net_credit: float,
        option_type: OptionType
    ) -> Tuple[float, float]:
        """
        Calculate breakeven points for a credit spread.
        
        Args:
            short_leg: Short leg contract
            long_leg: Long leg contract
            net_credit: Net credit received
            option_type: Type of spread (CALL or PUT)
            
        Returns:
            Tuple of (lower_breakeven, upper_breakeven)
        """
        if option_type == OptionType.CALL:
            # Call credit spread: breakeven at short_strike + net_credit
            breakeven = float(short_leg.strike_price.value) + net_credit
            return breakeven, breakeven
        else:  # PUT
            # Put credit spread: breakeven at short_strike - net_credit
            breakeven = float(short_leg.strike_price.value) - net_credit
            return breakeven, breakeven
    
    @staticmethod
    def find_weekly_expirations(
        contracts: List[OptionContractDTO]
    ) -> List[str]:
        """
        Find all weekly expiration dates (Fridays).
        
        Args:
            contracts: List of option contracts
            
        Returns:
            List of weekly expiration date strings
        """
        weekly_expirations = []
        
        for contract in contracts:
            if contract.expiration_date.is_weekly():
                exp_str = str(contract.expiration_date)
                if exp_str not in weekly_expirations:
                    weekly_expirations.append(exp_str)
        
        return sorted(weekly_expirations)
    
    @staticmethod
    def find_monthly_expirations(
        contracts: List[OptionContractDTO]
    ) -> List[str]:
        """
        Find all monthly expiration dates (third Friday of month).
        
        Args:
            contracts: List of option contracts
            
        Returns:
            List of monthly expiration date strings
        """
        monthly_expirations = []
        
        for contract in contracts:
            if contract.expiration_date.is_monthly():
                exp_str = str(contract.expiration_date)
                if exp_str not in monthly_expirations:
                    monthly_expirations.append(exp_str)
        
        return sorted(monthly_expirations)
    
    @staticmethod
    def find_credit_spread_max_credit_width(
        contracts: List[OptionContractDTO],
        current_price: float,
        expiration: str,
        get_bar_fn: Callable[[OptionContractDTO, datetime], Optional[OptionBarDTO]],
        date: datetime,
        min_spread_width: int = 4,
        max_spread_width: int = 10,
        max_strike_difference: float = 2.0,
        option_type: OptionType = OptionType.PUT
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best credit spread from a list of contracts that maximizes credit/width ratio.
        
        Evaluates spreads from min_spread_width to max_spread_width and selects the one
        with the highest credit/width ratio.
        
        Args:
            contracts: List of option contracts to evaluate
            current_price: Current underlying price
            expiration: Target expiration date string (YYYY-MM-DD)
            get_bar_fn: Function to get bar data: (contract: OptionContractDTO, date: datetime) -> Optional[OptionBarDTO]
            date: Date to get bar data for
            min_spread_width: Minimum spread width to evaluate (default: 4)
            max_spread_width: Maximum spread width to evaluate (default: 10)
            max_strike_difference: Maximum acceptable difference from target strike (default: 2.0)
            option_type: Option type to use (PUT or CALL, default: PUT)
            
        Returns:
            Dict with keys:
                - 'atm_contract': OptionContractDTO for ATM leg
                - 'otm_contract': OptionContractDTO for OTM leg
                - 'atm_bar': OptionBarDTO for ATM leg
                - 'otm_bar': OptionBarDTO for OTM leg
                - 'credit': float, net credit received
                - 'width': float, actual spread width
                - 'credit_width_ratio': float, credit/width ratio
            Or None if no valid spread found
        """
        # Filter contracts for the target expiration
        contracts_for_expiration = [
            c for c in contracts 
            if str(c.expiration_date) == expiration and c.contract_type == option_type
        ]
        
        if not contracts_for_expiration:
            return None
        
        # Find ATM contract
        atm_strike = round(current_price)
        atm_contract = None
        
        if option_type == OptionType.PUT:
            _, atm_contract = OptionsRetrieverHelper.find_atm_contracts(contracts_for_expiration, current_price)
        else:  # CALL
            atm_contract, _ = OptionsRetrieverHelper.find_atm_contracts(contracts_for_expiration, current_price)
        
        if not atm_contract:
            return None
        
        best_spread = None
        best_credit_width_ratio = -1.0
        
        # Evaluate spreads from min to max width
        for spread_width in range(min_spread_width, max_spread_width + 1):
            if option_type == OptionType.PUT:
                target_otm_strike = atm_strike - spread_width
            else:  # CALL
                target_otm_strike = atm_strike + spread_width
            
            # Find the contract with the closest strike to the target OTM strike
            otm_contract = min(
                contracts_for_expiration,
                key=lambda c: abs(float(c.strike_price.value) - target_otm_strike)
            )
            
            strike_difference = abs(float(otm_contract.strike_price.value) - target_otm_strike)
            
            # Skip if strike is too far from target
            if strike_difference > max_strike_difference:
                continue
            
            # Verify both legs have the same expiration (vertical spread check)
            if str(atm_contract.expiration_date) != str(otm_contract.expiration_date):
                continue
            
            # Get bar data to calculate net credit
            atm_bar = get_bar_fn(atm_contract, date)
            otm_bar = get_bar_fn(otm_contract, date)
            
            if not atm_bar or not otm_bar:
                continue
            
            # Calculate net credit (sell ATM, buy OTM)
            net_credit = float(atm_bar.close_price) - float(otm_bar.close_price)
            
            if net_credit <= 0:
                continue
            
            # Calculate actual spread width and credit/width ratio
            actual_spread_width = abs(float(atm_contract.strike_price.value) - float(otm_contract.strike_price.value))
            credit_width_ratio = net_credit / actual_spread_width
            
            # Track the best spread (highest credit/width ratio)
            if credit_width_ratio > best_credit_width_ratio:
                best_credit_width_ratio = credit_width_ratio
                best_spread = {
                    'atm_contract': atm_contract,
                    'otm_contract': otm_contract,
                    'atm_bar': atm_bar,
                    'otm_bar': otm_bar,
                    'credit': net_credit,
                    'width': actual_spread_width,
                    'credit_width_ratio': credit_width_ratio
                }
        
        return best_spread
    
    @staticmethod
    def find_debit_spread_max_reward_risk(
        contracts: List[OptionContractDTO],
        current_price: float,
        expiration: str,
        get_bar_fn: Callable[[OptionContractDTO, datetime], Optional[OptionBarDTO]],
        date: datetime,
        min_spread_width: int = 4,
        max_spread_width: int = 10,
        max_strike_difference: float = 2.0,
        option_type: OptionType = OptionType.CALL
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best debit spread from a list of contracts that maximizes reward/risk ratio.
        
        Evaluates spreads from min_spread_width to max_spread_width and selects the one
        with the highest reward/risk ratio.
        
        For a debit spread:
        - Buy the ITM/ATM leg (pay premium)
        - Sell the OTM leg (receive premium)
        - Net debit = ITM price - OTM price (what you pay)
        - Max profit = spread width - net debit
        - Max loss = net debit
        - Reward/Risk ratio = (spread width - net debit) / net debit
        
        Args:
            contracts: List of option contracts to evaluate
            current_price: Current underlying price
            expiration: Target expiration date string (YYYY-MM-DD)
            get_bar_fn: Function to get bar data: (contract: OptionContractDTO, date: datetime) -> Optional[OptionBarDTO]
            date: Date to get bar data for
            min_spread_width: Minimum spread width to evaluate (default: 4)
            max_spread_width: Maximum spread width to evaluate (default: 10)
            max_strike_difference: Maximum acceptable difference from target strike (default: 2.0)
            option_type: Option type to use (PUT or CALL, default: CALL)
            
        Returns:
            Dict with keys:
                - 'itm_contract': OptionContractDTO for ITM/ATM leg (long)
                - 'otm_contract': OptionContractDTO for OTM leg (short)
                - 'itm_bar': OptionBarDTO for ITM/ATM leg
                - 'otm_bar': OptionBarDTO for OTM leg
                - 'debit': float, net debit paid
                - 'width': float, actual spread width
                - 'max_profit': float, maximum potential profit
                - 'max_loss': float, maximum potential loss (the debit)
                - 'reward_risk_ratio': float, reward/risk ratio
            Or None if no valid spread found
        """
        # Filter contracts for the target expiration
        contracts_for_expiration = [
            c for c in contracts 
            if str(c.expiration_date) == expiration and c.contract_type == option_type
        ]
        
        if not contracts_for_expiration:
            return None
        
        # Find ATM contract (we'll use this as the long leg)
        atm_strike = round(current_price)
        itm_contract = None
        
        if option_type == OptionType.PUT:
            _, itm_contract = OptionsRetrieverHelper.find_atm_contracts(contracts_for_expiration, current_price)
        else:  # CALL
            itm_contract, _ = OptionsRetrieverHelper.find_atm_contracts(contracts_for_expiration, current_price)
        
        if not itm_contract:
            return None
        
        best_spread = None
        best_reward_risk_ratio = -1.0
        
        # Evaluate spreads from min to max width
        for spread_width in range(min_spread_width, max_spread_width + 1):
            if option_type == OptionType.PUT:
                # For put debit spread: buy higher strike (ATM/ITM), sell lower strike (OTM)
                target_otm_strike = atm_strike - spread_width
            else:  # CALL
                # For call debit spread: buy lower strike (ATM/ITM), sell higher strike (OTM)
                target_otm_strike = atm_strike + spread_width
            
            # Find the contract with the closest strike to the target OTM strike
            otm_contract = min(
                contracts_for_expiration,
                key=lambda c: abs(float(c.strike_price.value) - target_otm_strike)
            )
            
            strike_difference = abs(float(otm_contract.strike_price.value) - target_otm_strike)
            
            # Skip if strike is too far from target
            if strike_difference > max_strike_difference:
                continue
            
            # Verify both legs have the same expiration (vertical spread check)
            if str(itm_contract.expiration_date) != str(otm_contract.expiration_date):
                continue
            
            # Get bar data to calculate net debit
            itm_bar = get_bar_fn(itm_contract, date)
            otm_bar = get_bar_fn(otm_contract, date)
            
            if not itm_bar or not otm_bar:
                continue
            
            # Calculate net debit (buy ITM/ATM, sell OTM)
            net_debit = float(itm_bar.close_price) - float(otm_bar.close_price)
            
            # Debit must be positive (we pay to enter the position)
            if net_debit <= 0:
                continue
            
            # Calculate actual spread width
            actual_spread_width = abs(float(itm_contract.strike_price.value) - float(otm_contract.strike_price.value))
            
            # Calculate max profit and reward/risk ratio
            max_profit = actual_spread_width - net_debit
            max_loss = net_debit
            
            # Skip if max profit is negative or zero
            if max_profit <= 0:
                continue
            
            reward_risk_ratio = max_profit / max_loss
            
            # Track the best spread (highest reward/risk ratio)
            if reward_risk_ratio > best_reward_risk_ratio:
                best_reward_risk_ratio = reward_risk_ratio
                best_spread = {
                    'itm_contract': itm_contract,
                    'otm_contract': otm_contract,
                    'itm_bar': itm_bar,
                    'otm_bar': otm_bar,
                    'debit': net_debit,
                    'width': actual_spread_width,
                    'max_profit': max_profit,
                    'max_loss': max_loss,
                    'reward_risk_ratio': reward_risk_ratio
                }
        
        return best_spread
    
