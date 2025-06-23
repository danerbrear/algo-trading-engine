#!/usr/bin/env python3
"""
Setup script to create a virtual environment and .env file for environment variables.
"""
import os
import subprocess
import sys
from pathlib import Path

def create_venv():
    """Create a virtual environment if it doesn't exist"""
    venv_path = Path('venv')
    requirements_path = Path('requirements.txt')
    
    if not requirements_path.exists():
        print("Error: requirements.txt not found!")
        sys.exit(1)
    
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        
        # Get the pip path in the virtual environment
        if os.name == 'nt':  # Windows
            pip_path = venv_path / 'Scripts' / 'pip'
        else:  # Unix/MacOS
            pip_path = venv_path / 'bin' / 'pip'
            
        # Install required packages
        print("Installing required packages...")
        subprocess.run([str(pip_path), 'install', '-U', 'pip'], check=True)
        subprocess.run([str(pip_path), 'install', '-r', 'requirements.txt'], check=True)
        print("All required packages installed successfully!")
    else:
        print("Virtual environment already exists")
        update = input("Would you like to update packages? (y/n): ").strip().lower()
        if update == 'y':
            if os.name == 'nt':  # Windows
                pip_path = venv_path / 'Scripts' / 'pip'
            else:  # Unix/MacOS
                pip_path = venv_path / 'bin' / 'pip'
            print("Updating packages...")
            subprocess.run([str(pip_path), 'install', '-U', '-r', 'requirements.txt'], check=True)
            print("Packages updated successfully!")

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
    print("Setting up development environment...\n")
    
    try:
        create_venv()
        setup_env_file()
        
        print("\nSetup completed successfully!")
        print("\nTo activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print("    Run: .\\venv\\Scripts\\activate")
        else:  # Unix/MacOS
            print("    Run: source venv/bin/activate")
            
    except Exception as e:
        print(f"\nError during setup: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 