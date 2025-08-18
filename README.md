# GSoC 2025: Cloud-Native Tools for Detecting Insecticide Resistance in Malaria Mosquitoes

**Organization**: Wellcome Sanger, Tree of Life
**Project**: Google Summer of Code 2025

**Contributor**: Mohamed Laarej

**Mentors**: Jon Brenas, Anastasia HK, Chris Clarkson

## Project Overview

This repository contains the implementation of statistical and cloud-native tools to enhance the detection of novel insecticide resistance in *Anopheles* mosquito populations, as part of the Wellcome Sanger, Tree of Life GSoC 2025 project.

The tools aim to go beyond standard PLINK-based GWAS by integrating:

- Mixed-effects and Bayesian statistical models  
- Population structure correction  
- Genome-wide scanning pipelines  
- Interactive and cloud-optimized visualization components

## Project Goals

- Improve sensitivity and specificity in detecting resistance-associated variants  
- Enable analysis even when the phenotype labels are sparse or partially missing  
- Offer cloud-native infrastructure for global malaria researchers  
- Deliver reproducible, tested, and documented code modules  

## Repository Structure
```bash
gsoc-malaria-ir-detection/
├── data/ # Processed data and simulation results (excluded from Git)
├── notebooks/ # Exploratory and implementation notebooks
│ ├── 01_simulation.ipynb
│ ├── 02_mixed_effects_modeling.ipynb
│ └── ...
├── src/
│ ├── init.py
│ ├── data/ # Data loading & simulation
│ ├── models/ # Statistical model implementations (e.g., mixed models, Bayesian)
│ ├── viz/ # Visualization tools (plots, browser, etc.)
| ├── analysis/ # GWAS analyses and pipelines
│   └── gwas/
│       └── ...
│ └── utils/ # Shared helpers
├── tests/ # Unit tests for core modules
├── analysis/ # GWAS analyses and pipelines
│   └── gwas/
│       └── ...
├── poetry.lock # Dependency lockfile
├── pyproject.toml # Poetry dependency config
├── README.md
└── .gitignore
```

## Installation Instructions

### Prerequisites
- Python 3.9+
- Git
- Poetry (for dependency management): https://python-poetry.org/docs/

### Setup

```bash
# Clone the repository
git clone git@github.com:malariagen/vector_gwas_exploration.git
cd vector_gwas_exploration

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```
### Running Notebooks
```bash
jupyter notebook
```
### Running Tests
```bash
# Run all tests
poetry run pytest

# Run specific test
poetry run pytest tests/test_models.py
```

## Example Usage
```python
from src.models.mixed_effects import MixedEffectsModel
from src.data.loader import load_simulated_dataset

ds = load_simulated_dataset()
model = MixedEffectsModel().fit(ds)
model.summary()
```

## Progress Tracking

- [x] Repo initialization and Poetry setup
- [x] Simulated phenotype dataset
- [x] Mixed-effects modeling pipeline
- [ ] Bayesian modeling
- [ ] Population structure correction (PCA, etc.)
- [ ] Visualization (Manhattan plot, genome browser)
- [ ] Cloud packaging (Docker)
- [ ] Final evaluation and documentation
