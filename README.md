# AAVENUE: Detecting LLM Biases on NLU Tasks in AAVE via a Novel Benchmark

Welcome to the **AAVENUE** repository. This project introduces a benchmark that evaluates large language model (LLM) performance on natural language understanding (NLU) tasks in African American Vernacular English (AAVE) and Standard American English (SAE). This repository contains the code used to run experiments and analysis as described in our research paper.

## Overview
**AAVENUE** is designed to detect biases in LLMs across dialects by translating NLU tasks from SAE to AAVE and comparing the model performance on both. This benchmark highlights discrepancies that could lead to biased NLP systems and emphasizes the need for inclusivity in language technologies.

### Key Tasks:
- **BoolQ** (Boolean Question Answering)
- **MultiRC** (Multiple Sentence Reading Comprehension)
- **SST-2** (Sentiment Analysis)
- **COPA** (Causal Reasoning)
- **WSC** (Coreference Resolution)

## Project Structure
```bash
AAVENUE-Code/
│
├── BARTScores/               # Code for BARTScores evaluation
├── BLEU Score Code/          # Code for calculating BLEU scores
├── Comparison Scores Code/   # Code for comparing model translations
├── Determine Percentage/     # Code for determining performance percentage
├── Evaluation Code/          # Core evaluation scripts
├── Translation GPT 4o Mini Code/  # Code for GPT-4o mini translations
└── VALUE Translation/        # Code related to the VALUE benchmark translations
