# ALHD-experiments
This repository contains the code and scripts used to benchmark the ALHD (Arabic LLM and Human Dataset), a large-scale, multigenre, and comprehensive dataset for detecting Arabic LLM-generated texts.

## Dataset
The dataset is publicly available on Zenodo:
DOI: 10.5281/zenodo.17249602

## Licensing:
- ALHD Dataset: CC BY-NC 4.0 (Attribution-NonCommercial)
- Source datasets include SANAD, ANAD, MASC (CC BY 4.0), and HARD/BRAD (unspecified)
- Code is released under the MIT License (you are free to use, modify, and share the code with proper attribution)

## Code
This repository provides:
- Training and evaluation scripts for traditional ML models, transformer BERT and LLMs.
- Switchable configurations for different scenarios and datasets.
- Logging utilities for reproducibility.

## Usage examples
python ALHD_Traditional_Benchmarking.py --test_sources SANAD --data_size 10
python ALHD_Traditional_Benchmarking.py --test_sources SANAD --data_size full

## Citation
If you use this repository or the dataset, please cite:

ALHD: A Large-Scale and Multigenre Benchmark Dataset for Arabic LLM-Generated Text Detection
Ali Khairallah, Arkaitz Zubiaga, 2025
DOI: 10.5281/zenodo.17249602
