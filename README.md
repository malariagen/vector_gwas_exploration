# GSoC 2025: Cloud-Native Tools for Detecting Insecticide Resistance in Malaria Mosquitoes

| | |
| :--- | :--- |
| **Organization** | Wellcome Sanger Institute, Tree of Life |
| **Project** | Google Summer of Code 2025 |
| **Contributor** | [Mohamed Laarej](https://github.com/mohamed-laarej) |
| **Mentors** | Jon Brenas, Anastasia Hernandez-Koutoucheva, Chris Clarkson |
| **GSoC Report** | **[Final Report](https://github.com/mohamed-laarej/gsoc-2025-report/blob/main/index.md)** |

---

## Project Overview

This repository contains the complete toolkit developed during the Google Summer of Code 2025 for enhancing the detection of novel insecticide resistance in *Anopheles* mosquitoes. The project's core deliverable is a two-phase analytical pipeline designed to navigate the complexities of genomic data with strong population structure.

The pipeline consists of:
1.  A **sensitive GWAS scan** to cast a wide net and identify all potential resistance-associated SNPs.
2.  A **rigorous verification phase** using a suite of advanced statistical models (Logistic Regression, Mixed-Effects, and Bayesian) to filter out false positives caused by confounding.
3.  An **interactive visualization dashboard** built with Bokeh to allow researchers to intuitively explore the results from a genome-wide scale down to a single SNP.

A key scientific finding of this project was that confounding from population structure can produce spurious signals that are even stronger than true biological signals, which powerfully validates the necessity of this two-phase design.
 
## Repository Structure
```bash
gsoc-malaria-ir-detection/
├── data/ # Processed data and simulation results (excluded from Git)
├── notebooks/ # Exploratory and implementation notebooks
│ ├── 01_simulation.ipynb
│ ├── 02_mixed_effects_modeling.ipynb
│ └── ...
├── output/ # Generated HTML dashboards and other outputs
├── src/
│ ├── init.py
│ ├── data/ # Data loading & simulation
│ ├── models/ # Statistical model implementations (e.g., mixed models, Bayesian)
│ ├── viz/ # Dashboard builder script
| ├── analysis/ # GWAS analyses and pipelines
│    └── gwas/
│        └── ...
│ └── utils/ # Shared helpers
├── tests/ # Unit tests for core modules
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
## Example Usage

#### Step 1: Generate the mock data by running the notebook
```python
# (Inside the poetry shell, start Jupyter)
jupyter notebook
# -> Now, open and run `notebooks/08_generate_mock_data.ipynb`
```
#### Step 2: Build the standalone HTML dashboard
```python
# (From your terminal, still in the poetry shell)
python src/viz/build_explorer.py
```
#### Step 3: View the result
```python
# Open the newly created file in your web browser:
# output/gwas_explorer.html
```

## Progress Tracking

- [x] Phenotype Loading: Contributed functions to the main malariagen_data API.
- [x] Data Simulation: Built a ResistanceSimulator to create realistic test data with confounding.
- [x] Statistical Models: Implemented a suite of three verification models (Logistic, Mixed-Effects, Bayesian).
- [x] GWAS Scanner: Built and validated a robust Chi-squared scanner.
- [x] Validation: Performed rigorous positive and negative control tests, uncovering the strong effect of population structure.
- [x] CI/CD: Set up a GitHub Actions pipeline with black and ruff for code quality.
- [x] Interactive Dashboard: Built a complete, three-panel dashboard prototype with Bokeh.
- [ ] Full Genome Scan: The final computational run to generate the real dataset for the dashboard is the next step.
- [ ] Hierarchical Bayesian Model: Implementation of a more advanced model to control for confounding is a key piece of future work.
- [ ] Cloud packaging (Docker)
