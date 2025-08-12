# Cursor Rules for Project and File Structure

## Root Directory Structure

### Core Directory Organization
- **`src/`** - Main source code directory containing all Python packages
- **`data_cache/`** - Cached market data organized by data type
- **`predictions/`** - Output files for daily predictions
- **`venv/`** - Python virtual environment (excluded from git)
- **`__pycache__/`** - Python cache files (excluded from git)

### Configuration and Documentation
- **Root level config files** - Project-wide configuration and setup
- **Documentation files** - README, usage guides, and documentation

## src/ Package Structure

### Package Organization Principles
- Each folder in `src/` should be its own Python package
- Each package should have an `__init__.py` file
- Each package should have a `main.py` that can be run with `python -m src.<package>.main`
- Packages should be focused on a single domain or responsibility

### Package Structure Guidelines

#### 1. **model/** - Machine Learning Models
- **Purpose**: Core ML models and training logic
- **Structure**:
  - `main.py` - Entry point for training and evaluation
  - Model implementation files (one per model type)
  - Configuration files for model settings
  - Utility files for visualization and tracking
  - Feature engineering modules

#### 2. **common/** - Shared Utilities
- **Purpose**: Reusable components and utilities
- **Structure**:
  - Core utility modules
  - Shared data models and DTOs
  - Common functions and helpers
  - Subpackages for specialized functionality (e.g., `cache/`)

#### 3. **prediction/** - Prediction Pipeline
- **Purpose**: Making predictions on current market data
- **Structure**:
  - Main prediction script
  - Prediction-specific models and utilities
  - Configuration management
  - Documentation for prediction components

#### 4. **strategies/** - Trading Strategies
- **Purpose**: Implementation of trading strategies
- **Structure**:
  - One file per strategy implementation
  - Strategy-specific utilities and helpers
  - Common strategy interfaces or base classes

#### 5. **backtest/** - Backtesting Framework
- **Purpose**: Strategy backtesting and performance analysis
- **Structure**:
  - Main backtesting entry point
  - Backtesting data models and utilities
  - Performance analysis tools

## Data Directory Structure

### data_cache/ Organization
- **Organize by data type**: Each data type gets its own subdirectory
- **Examples**:
  - `stocks/` - Stock price data cache
  - `options/` - Options chain data cache
  - `treasury/` - Treasury yield data cache
  - `calendar/` - Calendar data cache
- **Structure within each subdirectory**:
  - Use consistent file naming patterns
  - Organize by date, ticker, or other logical grouping
  - Include metadata files for cache management

### predictions/ Organization
- **File naming convention**: Include ticker symbol and date
- **Structure**:
  - One file per prediction run
  - Consistent format across all prediction files
  - Include timestamp and metadata in filename

## Directory Organization Rules

### 1. **Single Responsibility per Directory**
- Each directory should have a clear, single purpose
- Avoid mixing different types of functionality in the same directory
- Use subdirectories to organize related functionality

### 2. **Logical Grouping**
- Group related files together in subdirectories
- Use descriptive directory names that indicate purpose
- Maintain consistent naming conventions across directories

### 3. **Hierarchical Organization**
- Use a clear hierarchy: root → main directories → subdirectories → files
- Keep the hierarchy shallow (max 3-4 levels deep)
- Use subdirectories to organize complex packages

### 4. **Scalability Considerations**
- Design directory structure to accommodate growth
- Use subdirectories to prevent directories from becoming too large
- Plan for future additions and modifications

## Package Dependencies and Relationships

### Dependency Flow
- **common/** - Base utilities used by all other packages
- **model/** - Depends on common/, provides ML functionality
- **prediction/** - Depends on model/ and common/, provides prediction pipeline
- **strategies/** - Depends on common/, implements trading strategies
- **backtest/** - Depends on strategies/ and common/, provides backtesting

### Import Structure Guidelines
- Use relative imports within packages
- Use absolute imports for cross-package dependencies
- Maintain clear dependency boundaries between packages

## Configuration and Environment Structure

### Configuration Organization
- **Root level**: Project-wide configuration files
- **Package level**: Package-specific configuration in each package
- **Environment**: Use `.env` files for sensitive data
- **Documentation**: Include configuration guides and examples

### Environment Setup
- **Virtual environment**: Use `venv/` for Python environment
- **Dependencies**: Use `requirements.txt` for dependency management
- **Setup scripts**: Include environment initialization scripts

## Data Management Structure

### Caching Strategy
- **Organize by data type**: Separate directories for different data types
- **Consistent naming**: Use consistent file naming patterns
- **Metadata management**: Include metadata files for cache management
- **Cleanup strategies**: Implement cache invalidation and cleanup

### Output Organization
- **Predictions**: Timestamped files with consistent naming
- **Results**: Organize by date, strategy, or other logical grouping
- **Logs**: Separate logging from output data

## Testing Structure

### Test Organization
- **Mirror package structure**: Create `tests/` directory that mirrors `src/` structure
- **Test naming**: Use `test_` prefix for test files
- **Test organization**: Group tests by functionality and package

## Documentation Structure

### Documentation Organization
- **Root level**: Main project documentation
- **Package level**: Package-specific documentation
- **Configuration**: Documentation for configuration options
- **Usage guides**: Step-by-step guides for common tasks

## Version Control Structure

### Git Organization
- **Ignore patterns**: Exclude virtual environments, cache files, and system files
- **Branch strategy**: Use feature branches for development
- **Commit organization**: Group related changes in logical commits

## Performance and Maintenance

### Directory Optimization
- **Keep directories focused**: Avoid overly large directories
- **Use subdirectories**: Break down large directories into logical subdirectories
- **Consistent patterns**: Use consistent organization patterns across the project

### Maintenance Considerations
- **Easy navigation**: Structure should be intuitive to navigate
- **Clear boundaries**: Clear separation between different types of functionality
- **Extensibility**: Structure should accommodate future additions

## Security and Best Practices

### Directory Security
- **Sensitive data**: Keep sensitive data in appropriate directories
- **Access control**: Use appropriate file permissions
- **Environment separation**: Separate development and production configurations

### Data Organization
- **Input validation**: Validate data at directory boundaries
- **Error handling**: Implement proper error handling for file operations
- **Backup strategies**: Plan for data backup and recovery