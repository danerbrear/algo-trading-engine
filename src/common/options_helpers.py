"""
Helper classes for options data retrieval and processing.

This module contains static utility methods for common options operations
as specified in features/improved_data_fetching.md Phase 2.
"""

from typing import List, Tuple, Optional
from decimal import Decimal

from .options_dtos import OptionContractDTO, StrikeRangeDTO, ExpirationRangeDTO
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
        tolerance = Decimal('0.01')  # 1 cent tolerance
        
        call_contract = None
        put_contract = None
        
        for contract in contracts:
            # Convert strike price to Decimal for comparison
            strike_decimal = Decimal(str(contract.strike_price.value))
            if abs(strike_decimal - current_decimal) <= tolerance:
                if contract.contract_type == OptionType.CALL:
                    call_contract = contract
                elif contract.contract_type == OptionType.PUT:
                    put_contract = contract
        
        return call_contract, put_contract
    
    @staticmethod
    def calculate_spread_width(
        short_leg: OptionContractDTO, 
        long_leg: OptionContractDTO
    ) -> float:
        """
        Calculate spread width between two contracts.
        
        Args:
            short_leg: Short leg contract
            long_leg: Long leg contract
            
        Returns:
            Spread width in dollars
        """
        return float(abs(short_leg.strike_price.value - long_leg.strike_price.value))
    
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
        Find contracts of specific type (call or put).
        
        Args:
            contracts: List of option contracts
            option_type: Option type to filter for
            
        Returns:
            List of contracts of specified type
        """
        return [contract for contract in contracts if contract.contract_type == option_type]
    
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
