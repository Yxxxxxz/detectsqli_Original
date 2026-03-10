# Hybrid SQL Injection Detector

Hybrid detection system combining:

- Signature Detection
- Fuzzy Matching
- Word2Vec Embedding
- RandomForest Classifier

## Architecture

Payload
 ↓
Data Cleaning
 ↓
Normalization
 ↓
Signature + Fuzzy Detection
 ↓
Word2Vec Embedding
 ↓
RandomForest Classification

## Installation

```bash
pip install -r requirements.txt