# EvaluacionQPP - Evaluation of Query Performance Prediction Methods

## Description
This project implements and evaluates various Query Performance Prediction (QPP) methods on document collections. QPP methods attempt to predict how well a query will perform before executing it.

## Key Features
- ✨ Implementation of pre-retrieval and post-retrieval QPP methods
- 📚 Support for multiple datasets (e.g., ANTIQUE, a local dataset [ANONYMIZED])
- 📊 Correlation analysis with evaluation metrics (e.g., nDCG, AP)
- 📈 Generation of charts and result reports
- ⚙️ Configurable processing of queries and documents

## Implemented QPP Methods
### Pre-retrieval:
- Average and maximum IDF
- Average and maximum SCQ
- Clarity Score

### Post-retrieval:
- WIG (Weighted Information Gain)
- NQC (Normalized Query Commitment)
- UEF (Utility Estimation Framework)

## Requirements
- Python 3.9+
- Java 11+
- Dependencies listed in `requirements.txt`

## Installation
1. Clone the repository
2. Install Java 11
3. Create a virtual environment:
   ```bash
   python -m venv qppenv
   ```
4. Activate the virtual environment:
   - **Windows**:
     ```cmd
     qppenv\Scripts\activate
     ```
   - **Unix/MacOS**:
     ```bash
     source qppenv/bin/activate
     ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Datasets
Most datasets are retrieved from [ir-datasets](https://ir-datasets.com/) (identifiers prefixed with `irds:` in `utils/config.py`) and are automatically downloaded on the first run via PyTerrier/ir_datasets. The download time may vary depending on the size of the corpus (e.g., TREC-COVID and MSMARCO v2 are large). The local dataset `[ANONYMIZED]_small` is stored locally within the project.

## Usage
All commands must be executed from the `EvaluacionQPP/` directory (where `main.py` is located).

### Run evaluation on a specific dataset
```bash
python main.py --datasets cranfield
```
Replace `cranfield` with any available dataset defined in `utils/config.py` (`AVAILABLE_DATASETS`): `antique_test`, `[ANONYMIZED]_small`, `cranfield`, `trec_covid`, `msmarco_dl20_judged`, `car_v15_trec_y1_manual`.

### Run evaluation on all datasets
```bash
python main.py
```
Omitting the `--datasets` flag processes all datasets defined in `AVAILABLE_DATASETS`.

### Main Options:
| Option | Description |
|--------|-------------|
| `--datasets` | Datasets to evaluate (space-separated) |
| `--max-queries` | Maximum number of queries to process |
| `--list-size` | List size for ranking metrics |
| `--metrics` | Evaluation metrics to use |
| `--correlations` | Correlation coefficients to calculate |
| `--output-dir` | Directory to save results |
| `--use-uef` | Include UEF method in evaluation |
| `--skip-plots` | Skip chart generation |

## Project Structure
```text
/EvaluacionQPP
  /data - Dataset management
  /indexing - Index building
  /metodos - QPP methods implementation
  /retrieval - Retrieval functions
  /utils - General utilities
  /correlation_analysis - Results analysis
```

## Contributions
Contributions are welcome. Please follow these style guidelines:
- Use `snake_case` for variables and functions
- Use `camelCase` for classes
- Use `ALL_CAPS` for constants
- Follow OOP principles
- Keep code modular and reusable
- Include documentation and comments
