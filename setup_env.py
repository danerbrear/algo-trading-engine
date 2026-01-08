#!/usr/bin/env python3
"""
Setup script to create a virtual environment and .env file for environment variables.

This script will:
1. Create a Python virtual environment
2. Install the algo-trading-engine package in editable mode
3. Install all dependencies from pyproject.toml
4. Set up the .env file with API keys
5. Make CLI commands available (algo-backtest, algo-paper-trade)

Requirements:
- Python 3.10 or higher (for yfinance 1.0+ compatibility)
- Polygon.io API key (for market data)

Usage:
    python3 setup_env.py
"""
import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version meets minimum requirements"""
    if sys.version_info < (3, 10):
        print("=" * 60)
        print("⚠️  WARNING: Python 3.10+ is required")
        print("=" * 60)
        print(f"Current version: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print("\nThis package requires Python 3.10 or higher for:")
        print("  - yfinance 1.0+ (better rate limit handling)")
        print("  - Modern Python syntax support")
        print("\nPlease upgrade Python or use Python 3.10+ to run this script:")
        print("  python3.10 setup_env.py")
        print("  # or")
        print("  python3.11 setup_env.py")
        print()
        sys.exit(1)
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected (meets requirements)")

def create_venv():
    """Create a virtual environment if it doesn't exist"""
    venv_path = Path('venv')
    pyproject_path = Path('pyproject.toml')
    
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found!")
        sys.exit(1)
    
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        
        # Get the python and pip paths in the virtual environment
        if os.name == 'nt':  # Windows
            python_path = venv_path / 'Scripts' / 'python.exe'
            pip_path = venv_path / 'Scripts' / 'pip.exe'
        else:  # Unix/MacOS
            python_path = venv_path / 'bin' / 'python'
            pip_path = venv_path / 'bin' / 'pip'
        
        # Check Python version
        result = subprocess.run([str(python_path), '--version'], capture_output=True, text=True)
        python_version = result.stdout.strip()
        print(f"Using {python_version}")
        
        # Upgrade pip using the python module approach (works on both platforms)
        print("Upgrading pip...")
        try:
            subprocess.run([str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not upgrade pip: {e}")
            print("Continuing with existing pip version...")
        
        # Install package in editable mode (this installs all dependencies from pyproject.toml)
        print("Installing algo-trading-engine package and dependencies...")
        subprocess.run([str(pip_path), 'install', '-e', '.'], check=True)
        print("✅ Package and dependencies installed successfully!")
        print("✅ CLI commands available: algo-backtest, algo-paper-trade")
    else:
        print("Virtual environment already exists")
        update = input("Would you like to reinstall/update packages? (y/n): ").strip().lower()
        if update == 'y':
            if os.name == 'nt':  # Windows
                python_path = venv_path / 'Scripts' / 'python.exe'
                pip_path = venv_path / 'Scripts' / 'pip.exe'
            else:  # Unix/MacOS
                python_path = venv_path / 'bin' / 'python'
                pip_path = venv_path / 'bin' / 'pip'
            
            print("Updating packages...")
            
            # Upgrade pip first
            try:
                subprocess.run([str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not upgrade pip: {e}")
            
            # Reinstall package (updates dependencies)
            subprocess.run([str(pip_path), 'install', '-e', '.', '--upgrade'], check=True)
            print("✅ Packages updated successfully!")

def setup_env_file():
    """Create or update the .env file with required API keys and model save path"""
    env_path = Path('.env')
    default_model_path = 'Trained_Models'
    
    if not env_path.exists():
        print("\nSetting up environment variables...")
        polygon_key = input("Enter your Polygon.io API key: ").strip()
        model_path = input(f"Enter model save base path (press Enter for default: {default_model_path}): ").strip()
        if not model_path:
            model_path = default_model_path
        with open(env_path, 'w') as f:
            f.write(f"POLYGON_API_KEY={polygon_key}\n")
            f.write(f"MODEL_SAVE_BASE_PATH={model_path}\n")
        print(".env file created successfully!")
    else:
        print("\n.env file already exists")
        update = input("Would you like to update the API keys or model path? (y/n): ").strip().lower()
        if update == 'y':
            # Read existing env file
            with open(env_path, 'r') as f:
                env_vars = dict(line.strip().split('=', 1) for line in f if line.strip() and not line.startswith('#'))
            # Update API keys
            polygon_key = input("Enter your Polygon.io API key (press Enter to keep existing): ").strip()
            if polygon_key:
                env_vars['POLYGON_API_KEY'] = polygon_key
            model_path = input(f"Enter model save base path (press Enter to keep existing: {env_vars.get('MODEL_SAVE_BASE_PATH', default_model_path)}): ").strip()
            if model_path:
                env_vars['MODEL_SAVE_BASE_PATH'] = model_path
            # Write back to env file
            with open(env_path, 'w') as f:
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
            print(".env file updated successfully!")

def main():
    """Main setup function"""
    print("=" * 60)
    print("Algo Trading Engine - Setup")
    print("=" * 60)
    print()
    
    try:
        check_python_version()
        create_venv()
        setup_env_file()
        
        print("\n" + "=" * 60)
        print("✅ Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print("   .\\venv\\Scripts\\activate")
        else:  # Unix/MacOS
            print("   source venv/bin/activate")
        print("\n2. Verify installation:")
        print("   python -c \"import algo_trading_engine; print(algo_trading_engine.__version__)\"")
        print("\n3. Run a backtest:")
        print("   algo-backtest --strategy velocity_momentum --symbol SPY")
        print("\n4. Or run paper trading:")
        print("   algo-paper-trade --strategy velocity_momentum --symbol SPY")
        print()
            
    except Exception as e:
        print(f"\n❌ Error during setup: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 