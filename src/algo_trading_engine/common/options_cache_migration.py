"""
Cache migration utility for OptionsHandler refactoring.

This module migrates existing options cache data to the new structure
specified in features/improved_data_fetching.md Phase 2.
"""

import os
import pickle
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from .cache.options_cache_manager import OptionsCacheManager
from algo_trading_engine.dto import OptionContractDTO, OptionBarDTO
from algo_trading_engine.vo import StrikePrice, ExpirationDate
from .models import OptionType, Option, OptionChain


class OptionsCacheMigrator:
    """
    Migrates existing options cache data to the new structure.
    
    Old Structure:
    data_cache/options/{symbol}/
    ├── {date}_contracts.pkl
    ├── {date}.pkl
    └── ...
    
    New Structure:
    data_cache/options/{symbol}/
    ├── {date}/
    │   ├── contracts.pkl
    │   └── bars/
    │       └── {ticker}.pkl
    """
    
    def __init__(self, base_dir: str = 'data_cache'):
        self.base_dir = Path(base_dir)
        self.new_cache_manager = OptionsCacheManager(base_dir)
    
    def discover_old_cache_files(self, symbol: str) -> Dict[date, Dict[str, Path]]:
        """
        Discover old cache files for a symbol.
        
        Returns:
            Dict mapping date to dict of file types and paths
        """
        symbol_dir = self.base_dir / 'options' / symbol
        if not symbol_dir.exists():
            return {}
        
        cache_files = {}
        
        for file_path in symbol_dir.glob('*.pkl'):
            filename = file_path.name
            
            # Parse date from filename
            date_str = None
            file_type = None
            
            if '_contracts.pkl' in filename:
                date_str = filename.replace('_contracts.pkl', '')
                file_type = 'contracts'
            elif filename.endswith('.pkl') and '_' not in filename:
                date_str = filename.replace('.pkl', '')
                file_type = 'chain'
            
            if date_str and file_type:
                try:
                    cache_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    
                    if cache_date not in cache_files:
                        cache_files[cache_date] = {}
                    
                    cache_files[cache_date][file_type] = file_path
                except ValueError:
                    print(f"Could not parse date from filename: {filename}")
                    continue
        
        return cache_files
    
    def migrate_contracts_file(self, old_file_path: Path, symbol: str, cache_date: date) -> bool:
        """
        Migrate a contracts file to the new format.
        
        Args:
            old_file_path: Path to the old contracts file
            symbol: Symbol name
            cache_date: Date of the cache
            
        Returns:
            True if migration successful, False otherwise
        """
        try:
            with open(old_file_path, 'rb') as f:
                old_data = pickle.load(f)
            
            # Convert old format to new DTOs
            new_contracts = []
            
            if isinstance(old_data, list):
                for item in old_data:
                    if isinstance(item, dict):
                        contract = self._convert_dict_to_contract_dto(item)
                        if contract:
                            new_contracts.append(contract)
                    elif hasattr(item, 'ticker'):  # Old Option object
                        contract = self._convert_option_to_contract_dto(item)
                        if contract:
                            new_contracts.append(contract)
            
            # Save to new format
            if new_contracts:
                self.new_cache_manager.save_contracts(symbol, cache_date, new_contracts)
                print(f"Migrated {len(new_contracts)} contracts for {symbol} on {cache_date}")
                return True
            
        except Exception as e:
            print(f"Error migrating contracts file {old_file_path}: {e}")
        
        return False
    
    def migrate_chain_file(self, old_file_path: Path, symbol: str, cache_date: date) -> bool:
        """
        Migrate a chain file to the new format.
        
        Args:
            old_file_path: Path to the old chain file
            symbol: Symbol name
            cache_date: Date of the cache
            
        Returns:
            True if migration successful, False otherwise
        """
        try:
            with open(old_file_path, 'rb') as f:
                old_data = pickle.load(f)
            
            # Handle different old data formats
            if isinstance(old_data, OptionChain):
                return self._migrate_option_chain(old_data, symbol, cache_date)
            elif isinstance(old_data, dict):
                return self._migrate_chain_dict(old_data, symbol, cache_date)
            elif isinstance(old_data, list):
                return self._migrate_chain_list(old_data, symbol, cache_date)
            
        except Exception as e:
            print(f"Error migrating chain file {old_file_path}: {e}")
        
        return False
    
    def _migrate_option_chain(self, option_chain: OptionChain, symbol: str, cache_date: date) -> bool:
        """Migrate an OptionChain object to new format."""
        contracts = []
        bars = {}
        
        # Convert calls
        for call in option_chain.calls:
            contract = self._convert_option_to_contract_dto(call)
            if contract:
                contracts.append(contract)
                
                # Create bar data if available
                bar = self._create_bar_from_option(call, cache_date)
                if bar:
                    bars[call.ticker] = bar
        
        # Convert puts
        for put in option_chain.puts:
            contract = self._convert_option_to_contract_dto(put)
            if contract:
                contracts.append(contract)
                
                # Create bar data if available
                bar = self._create_bar_from_option(put, cache_date)
                if bar:
                    bars[put.ticker] = bar
        
        # Save contracts
        if contracts:
            self.new_cache_manager.save_contracts(symbol, cache_date, contracts)
        
        # Save bars
        for ticker, bar in bars.items():
            self.new_cache_manager.save_bar(symbol, cache_date, ticker, bar)
        
        print(f"Migrated OptionChain: {len(contracts)} contracts, {len(bars)} bars for {symbol} on {cache_date}")
        return True
    
    def _migrate_chain_dict(self, chain_dict: Dict[str, Any], symbol: str, cache_date: date) -> bool:
        """Migrate a chain dictionary to new format."""
        contracts = []
        bars = {}
        
        # Handle different dictionary structures
        if 'calls' in chain_dict and 'puts' in chain_dict:
            # Standard chain format
            for call_data in chain_dict.get('calls', []):
                contract = self._convert_dict_to_contract_dto(call_data)
                if contract:
                    contracts.append(contract)
            
            for put_data in chain_dict.get('puts', []):
                contract = self._convert_dict_to_contract_dto(put_data)
                if contract:
                    contracts.append(contract)
        
        elif isinstance(chain_dict, dict) and any('strike' in str(v) for v in chain_dict.values()):
            # List of options in dictionary format
            for option_data in chain_dict.values():
                if isinstance(option_data, dict):
                    contract = self._convert_dict_to_contract_dto(option_data)
                    if contract:
                        contracts.append(contract)
        
        # Save contracts
        if contracts:
            self.new_cache_manager.save_contracts(symbol, cache_date, contracts)
            print(f"Migrated chain dict: {len(contracts)} contracts for {symbol} on {cache_date}")
            return True
        
        return False
    
    def _migrate_chain_list(self, chain_list: List[Any], symbol: str, cache_date: date) -> bool:
        """Migrate a chain list to new format."""
        contracts = []
        
        for item in chain_list:
            if isinstance(item, dict):
                contract = self._convert_dict_to_contract_dto(item)
                if contract:
                    contracts.append(contract)
            elif hasattr(item, 'ticker'):  # Old Option object
                contract = self._convert_option_to_contract_dto(item)
                if contract:
                    contracts.append(contract)
        
        # Save contracts
        if contracts:
            self.new_cache_manager.save_contracts(symbol, cache_date, contracts)
            print(f"Migrated chain list: {len(contracts)} contracts for {symbol} on {cache_date}")
            return True
        
        return False
    
    def _convert_option_to_contract_dto(self, option: Option) -> Optional[OptionContractDTO]:
        """Convert old Option object to new OptionContractDTO."""
        try:
            # Handle different option object types
            if hasattr(option, 'expiration_date'):
                # OptionsContract object (from Polygon API)
                exp_date_str = option.expiration_date
                underlying_ticker = option.underlying_ticker
                strike = option.strike_price
                option_type = option.contract_type
                exercise_style = option.exercise_style
                shares_per_contract = option.shares_per_contract
                primary_exchange = option.primary_exchange
                cfi = option.cfi
                additional_underlyings = option.additional_underlyings
            elif hasattr(option, 'expiration'):
                # Old Option object
                exp_date_str = option.expiration
                underlying_ticker = option.symbol
                strike = option.strike
                option_type = option.option_type
                exercise_style = "american"  # Default
                shares_per_contract = 100    # Default
                primary_exchange = None
                cfi = None
                additional_underlyings = None
            else:
                print(f"Unknown option object type: {type(option)}")
                return None
            
            # Parse expiration date (allow past dates for migration)
            exp_date_obj = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
            
            # For migration, we need to allow past dates, so we'll create the ExpirationDate
            # by temporarily bypassing the validation using object.__setattr__
            from algo_trading_engine.vo import ExpirationDate
            exp_date = ExpirationDate.__new__(ExpirationDate)
            object.__setattr__(exp_date, 'date', exp_date_obj)
            
            # Create strike price
            strike_price = StrikePrice(strike)
            
            # Determine contract type
            if isinstance(option_type, str):
                contract_type = OptionType.CALL if option_type.lower() == 'call' else OptionType.PUT
            else:
                contract_type = OptionType.CALL if option_type == OptionType.CALL else OptionType.PUT
            
            return OptionContractDTO(
                ticker=option.ticker,
                underlying_ticker=underlying_ticker,
                contract_type=contract_type,
                strike_price=strike_price,
                expiration_date=exp_date,
                exercise_style=exercise_style,
                shares_per_contract=shares_per_contract,
                primary_exchange=primary_exchange,
                cfi=cfi,
                additional_underlyings=additional_underlyings
            )
        except Exception as e:
            print(f"Error converting Option to ContractDTO: {e}")
            return None
    
    def _convert_dict_to_contract_dto(self, data: Dict[str, Any]) -> Optional[OptionContractDTO]:
        """Convert dictionary data to new OptionContractDTO."""
        try:
            # Extract required fields
            ticker = data.get('ticker', '')
            underlying_ticker = data.get('underlying_ticker', data.get('symbol', ''))
            contract_type_str = data.get('contract_type', data.get('option_type', ''))
            strike = data.get('strike_price', data.get('strike', 0))
            expiration_str = data.get('expiration_date', data.get('expiration', ''))
            
            if not all([ticker, underlying_ticker, contract_type_str, strike, expiration_str]):
                return None
            
            # Parse expiration date (allow past dates for migration)
            if isinstance(expiration_str, str):
                exp_date_obj = datetime.strptime(expiration_str, '%Y-%m-%d').date()
            else:
                exp_date_obj = expiration_str
            
            # For migration, we need to allow past dates, so we'll create the ExpirationDate
            # by temporarily bypassing the validation using object.__setattr__
            from algo_trading_engine.vo import ExpirationDate
            exp_date = ExpirationDate.__new__(ExpirationDate)
            object.__setattr__(exp_date, 'date', exp_date_obj)
            
            # Create strike price
            strike_price = StrikePrice(strike)
            
            # Determine contract type
            if contract_type_str.lower() in ['call', 'c']:
                contract_type = OptionType.CALL
            elif contract_type_str.lower() in ['put', 'p']:
                contract_type = OptionType.PUT
            else:
                return None
            
            return OptionContractDTO(
                ticker=ticker,
                underlying_ticker=underlying_ticker,
                contract_type=contract_type,
                strike_price=strike_price,
                expiration_date=exp_date,
                exercise_style=data.get('exercise_style', 'american'),
                shares_per_contract=data.get('shares_per_contract', 100),
                primary_exchange=data.get('primary_exchange'),
                cfi=data.get('cfi'),
                additional_underlyings=data.get('additional_underlyings')
            )
        except Exception as e:
            print(f"Error converting dict to ContractDTO: {e}")
            return None
    
    def _create_bar_from_option(self, option: Option, cache_date: date) -> Optional[OptionBarDTO]:
        """Create bar data from option if price/volume data is available."""
        try:
            if option.last_price is None:
                return None
            
            # Use current date as timestamp
            timestamp = datetime.combine(cache_date, datetime.min.time())
            
            # Create bar with available data
            return OptionBarDTO(
                ticker=option.ticker,
                timestamp=timestamp,
                open_price=option.last_price,
                high_price=option.last_price,
                low_price=option.last_price,
                close_price=option.last_price,
                volume=option.volume or 0,
                volume_weighted_avg_price=option.last_price,
                number_of_transactions=1,
                adjusted=True
            )
        except Exception as e:
            print(f"Error creating bar from option: {e}")
            return None
    
    def migrate_symbol(self, symbol: str, dry_run: bool = False) -> Dict[str, int]:
        """
        Migrate all cache files for a symbol.
        
        Args:
            symbol: Symbol to migrate
            dry_run: If True, only show what would be migrated
            
        Returns:
            Dict with migration statistics
        """
        print(f"Migrating cache for symbol: {symbol}")
        
        cache_files = self.discover_old_cache_files(symbol)
        if not cache_files:
            print(f"No old cache files found for {symbol}")
            return {'dates_processed': 0, 'contracts_migrated': 0, 'bars_migrated': 0}
        
        stats = {'dates_processed': 0, 'contracts_migrated': 0, 'bars_migrated': 0}
        
        for cache_date, files in cache_files.items():
            print(f"Processing {symbol} for date {cache_date}")
            
            if dry_run:
                print(f"  Would migrate: {list(files.keys())}")
                stats['dates_processed'] += 1
                continue
            
            # Migrate contracts file
            if 'contracts' in files:
                if self.migrate_contracts_file(files['contracts'], symbol, cache_date):
                    stats['contracts_migrated'] += 1
            
            # Migrate chain file
            if 'chain' in files:
                if self.migrate_chain_file(files['chain'], symbol, cache_date):
                    stats['bars_migrated'] += 1
            
            stats['dates_processed'] += 1
        
        return stats
    
    def migrate_all_symbols(self, dry_run: bool = False) -> Dict[str, Dict[str, int]]:
        """
        Migrate cache files for all symbols.
        
        Args:
            dry_run: If True, only show what would be migrated
            
        Returns:
            Dict mapping symbol to migration statistics
        """
        options_dir = self.base_dir / 'options'
        if not options_dir.exists():
            print("No options cache directory found")
            return {}
        
        all_stats = {}
        
        for symbol_dir in options_dir.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                stats = self.migrate_symbol(symbol, dry_run)
                all_stats[symbol] = stats
        
        return all_stats
    
    def cleanup_recent_option_bars(self, symbol: str, cutoff_date: date = date(2023, 10, 1), backup: bool = True) -> int:
        """
        Clean up option bar cache files more recent than cutoff date.
        
        This method specifically targets option bar files (chain files) that are newer
        than the specified cutoff date, while preserving contract files and older data.
        
        Args:
            symbol: Symbol to clean up
            cutoff_date: Delete files newer than this date (default: Oct 1, 2023)
            backup: If True, move files to backup directory instead of deleting
            
        Returns:
            Number of files cleaned up
        """
        symbol_dir = self.base_dir / 'options' / symbol
        if not symbol_dir.exists():
            return 0
        
        cleaned_count = 0
        backup_dir = self.base_dir / 'options' / symbol / 'backup'
        
        if backup:
            backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all chain files (option bar files) that are newer than cutoff
        for file_path in symbol_dir.glob('*.pkl'):
            filename = file_path.name
            
            # Skip contract files (they have '_contracts' in the name)
            if '_contracts.pkl' in filename:
                continue
            
            # Parse date from filename (format: YYYY-MM-DD.pkl)
            try:
                date_str = filename.replace('.pkl', '')
                file_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                # Only delete files newer than cutoff date
                if file_date > cutoff_date:
                    try:
                        if backup:
                            backup_path = backup_dir / filename
                            file_path.rename(backup_path)
                            print(f"🗑️  Backed up recent option bar file: {filename} (date: {file_date})")
                        else:
                            file_path.unlink()
                            print(f"🗑️  Deleted recent option bar file: {filename} (date: {file_date})")
                        
                        cleaned_count += 1
                    except Exception as e:
                        print(f"❌ Error cleaning up {filename}: {e}")
                        
            except ValueError:
                # Skip files that don't match the expected date format
                continue
        
        return cleaned_count
    
    def cleanup_old_files(self, symbol: str, backup: bool = True) -> int:
        """
        Clean up old cache files after successful migration.
        
        Args:
            symbol: Symbol to clean up
            backup: If True, move files to backup directory instead of deleting
            
        Returns:
            Number of files cleaned up
        """
        cache_files = self.discover_old_cache_files(symbol)
        if not cache_files:
            return 0
        
        cleaned_count = 0
        backup_dir = self.base_dir / 'options' / symbol / 'backup'
        
        if backup:
            backup_dir.mkdir(parents=True, exist_ok=True)
        
        for cache_date, files in cache_files.items():
            for file_type, file_path in files.items():
                try:
                    if backup:
                        backup_path = backup_dir / f"{cache_date}_{file_type}.pkl"
                        file_path.rename(backup_path)
                        print(f"Backed up {file_path} to {backup_path}")
                    else:
                        file_path.unlink()
                        print(f"Deleted {file_path}")
                    
                    cleaned_count += 1
                except Exception as e:
                    print(f"Error cleaning up {file_path}: {e}")
        
        return cleaned_count


def main():
    """Main function for running cache migration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate options cache to new format')
    parser.add_argument('--symbol', help='Symbol to migrate (default: all)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be migrated')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old files after migration')
    parser.add_argument('--cleanup-recent-bars', action='store_true', help='Clean up recent option bar files (newer than Oct 2023)')
    parser.add_argument('--backup', action='store_true', default=True, help='Backup old files instead of deleting')
    parser.add_argument('--cutoff-date', help='Cutoff date for recent bar cleanup (YYYY-MM-DD, default: 2023-10-01)')
    
    args = parser.parse_args()
    
    migrator = OptionsCacheMigrator()
    
    # Parse cutoff date if provided
    cutoff_date = date(2023, 10, 1)  # Default: Oct 1, 2023
    if args.cutoff_date:
        try:
            cutoff_date = datetime.strptime(args.cutoff_date, '%Y-%m-%d').date()
        except ValueError:
            print(f"❌ Invalid cutoff date format: {args.cutoff_date}. Use YYYY-MM-DD")
            return
    
    if args.symbol:
        # Migrate specific symbol
        stats = migrator.migrate_symbol(args.symbol, args.dry_run)
        print(f"Migration stats for {args.symbol}: {stats}")
        
        if args.cleanup_recent_bars and not args.dry_run:
            cleaned = migrator.cleanup_recent_option_bars(args.symbol, cutoff_date, args.backup)
            print(f"🗑️  Cleaned up {cleaned} recent option bar files (newer than {cutoff_date})")
        
        if args.cleanup and not args.dry_run:
            cleaned = migrator.cleanup_old_files(args.symbol, args.backup)
            print(f"Cleaned up {cleaned} old files")
    else:
        # Migrate all symbols
        all_stats = migrator.migrate_all_symbols(args.dry_run)
        print("Migration stats:")
        for symbol, stats in all_stats.items():
            print(f"  {symbol}: {stats}")
        
        if args.cleanup_recent_bars and not args.dry_run:
            total_cleaned = 0
            for symbol in all_stats.keys():
                cleaned = migrator.cleanup_recent_option_bars(symbol, cutoff_date, args.backup)
                total_cleaned += cleaned
                if cleaned > 0:
                    print(f"🗑️  Cleaned up {cleaned} recent option bar files for {symbol}")
            print(f"🗑️  Total recent option bar files cleaned: {total_cleaned}")
        
        if args.cleanup and not args.dry_run:
            for symbol in all_stats.keys():
                cleaned = migrator.cleanup_old_files(symbol, args.backup)
                print(f"Cleaned up {cleaned} old files for {symbol}")


if __name__ == '__main__':
    main()
