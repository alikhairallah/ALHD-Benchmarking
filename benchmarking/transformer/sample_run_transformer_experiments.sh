#!/bin/bash

# Activate your conda environment
source activate benchmarking-env

#on 10% subset
python ALHD_Transformer_Benchmarking.py --data_size 10

python ALHD_Transformer_Benchmarking.py --test_sources SANAD --data_size 10
python ALHD_Transformer_Benchmarking.py --test_sources ANAD --data_size 10
python ALHD_Transformer_Benchmarking.py --test_sources MDAT --data_size 10
python ALHD_Transformer_Benchmarking.py --test_sources MASC --data_size 10
python ALHD_Transformer_Benchmarking.py --test_sources HARD --data_size 10
python ALHD_Transformer_Benchmarking.py --test_sources BRAD --data_size 10

python ALHD_Transformer_Benchmarking.py --test_sources SANAD ANAD --data_size 10
python ALHD_Transformer_Benchmarking.py --test_sources MDAT MASC --data_size 10
python ALHD_Transformer_Benchmarking.py --test_sources HARD BRAD --data_size 10

#on full subset
python ALHD_Transformer_Benchmarking.py --data_size full

python ALHD_Transformer_Benchmarking.py --test_sources SANAD --data_size full
python ALHD_Transformer_Benchmarking.py --test_sources ANAD --data_size full
python ALHD_Transformer_Benchmarking.py --test_sources MDAT --data_size full
python ALHD_Transformer_Benchmarking.py --test_sources MASC --data_size full
python ALHD_Transformer_Benchmarking.py --test_sources HARD --data_size full
python ALHD_Transformer_Benchmarking.py --test_sources BRAD --data_size full

python ALHD_Transformer_Benchmarking.py --test_sources SANAD ANAD --data_size full
python ALHD_Transformer_Benchmarking.py --test_sources MDAT MASC --data_size full
python ALHD_Transformer_Benchmarking.py --test_sources HARD BRAD --data_size full


