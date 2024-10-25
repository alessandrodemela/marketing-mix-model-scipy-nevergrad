# Marketing Mix Modeling (MMM) Project

This repository contains a Python implementation of Marketing Mix Modeling, a statistical analysis method used to determine the effectiveness of marketing investments across various channels.

## Project Structure
```markdown
│   README.md
│   requirements.txt
│
├───config
│       params_sample.yaml
│
├───data
│       sample_data.csv
│
├───notebooks
│       analisys.ipynb
│
├───src
│   │   data_transformer.py
│   │   model_diagnostics.py
│   │   model_optimizer.py
│   │   __init__.py
│   │
│   ├───helper
│   │   │   helper.py
└───tests
        unit_tests.py
```

## Key Components

1. **Data Transformer (`data_transformer.py`)**  
   Handles data transformations essential for MMM:
   - Adstock transformation: Models the delayed effect of marketing activities.
   - Saturation effects: Models diminishing returns using Hill and Sigmoid functions.
   - Customizable decay rates and saturation parameters.

2. **Model Optimizer (`model_optimizer.py`)**  
   Core optimization module that:
   - Loads configuration from YAML files.
   - Optimizes model parameters using Nevergrad's CMA-ES algorithm.
   - Manages feature transformations and coefficient estimation.
   - Provides visualization tools for model results.

3. **Model Diagnostics (`model_diagnostics.py`)**  
   Comprehensive model evaluation tools:
   - Calculates key metrics (R², Adjusted R², RMSE, MAPE).
   - Performs regression diagnostics.
   - Generates confidence intervals.
   - Creates autocorrelation plots.

4. **Helper Functions (`helper.py`)**  
   Utility functions for:
   - Exploratory Data Analysis (EDA).
   - Marketing channel visualization.
   - Seasonality analysis.
   - Multicollinearity checks.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/marketing_mix_model_scipy_nevergrad.git
cd marketing_mix_model_scipy_nevergrad
```

1. Install required packages

```bash
pip install -r requirements.txt
```

## Usage

1. Configure your model parameters in ```config/params_sample.yaml```

2. Import Data

3. Run the model

```python
from src.model_optimizer import ModelOptimizer

# Initialize the optimizer
optimizer = ModelOptimizer("config/params_sample.yaml")

# Load and prepare your data
# data = ...

# Run optimization
optimizer.optimize_parameters(data)

# Get diagnostics
diagnostics = optimizer.get_diagnostics()
```

## Key Features

- **Flexible Data Transformations**: Support for multiple transformation types including adstock and saturation effects.
- **Advanced Optimization**: Uses Nevergrad's CMA-ES algorithm for robust parameter optimization.
- **Comprehensive Diagnostics**: Detailed model evaluation metrics and visualizations.
- **Configurable**: Easy parameter configuration through YAML files.
- **Visual Analysis**: Built-in visualization tools for model results and diagnostics.

## Dependencies

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `nevergrad`
- `matplotlib`
- `seaborn`
- `pyyaml`

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
