# Smart Product Pricing Solution – ML Challenge 2025

**Team Name:** Platypus  
**Team Members:** Purva Jivani, Laxman Patel, Mohd Haisam Khan, Mohak Arya  
**Submission Date:** 12-10-2025  

---

## Project Overview

This project predicts product prices using a **hybrid TF-IDF + LightGBM framework** that combines textual and structured data. The model leverages multi-level TF-IDF embeddings and feature engineering to capture product semantics, packaging cues, and numeric attributes for accurate pricing.

### Key Features
- Cleans and preprocesses product catalog text  
- Normalizes measurement units (fl, oz, bottle)  
- Extracts features like text length, number of digits, and numeric attributes  
- Uses multi-source TF-IDF embeddings (word, bigram, character n-grams)  
- Trains a LightGBM regression model with 5-fold cross-validation  
- Outputs robust price predictions with `log1p` transformation for stability  

---

## Methodology

### Problem Analysis
- **Task:** Supervised regression using catalog text and numeric attributes  
- **Challenges:**  
  - Mixed measurement units  
  - Missing values  
  - Variations in text length  

**Solutions Implemented:**  
- Text cleanup and normalization  
- Feature engineering for structured data  
- `log1p` transform for target stabilization  

### Solution Strategy
Hybrid approach: Combine text + structured features into a single LightGBM model.  

**Preprocessing Includes:**  
- Lowercasing, regex cleanup, stopword removal  

**TF-IDF Vectorization:**  
- Item Name: Word TF-IDF (2000 features)  
- Description: Word TF-IDF (5000 features)  
- Character n-grams (3–5, 2000 features)  

**Engineered Features:**  
- `Value`, `Unit_encoded`, `text_len`, `num_digits`, `num_tokens`  

**Workflow:**  
1. Concatenate all features  
2. Train LightGBM regression model  
3. Predict `log1p(price)` → inverse-transform → final price  

---

## Model Performance
- **SMAPE:** 52.39%  
- **MAE:** 11.59  

The hybrid TF-IDF + LightGBM approach provides stable predictions across catalog text and structured cues. Unit normalization and modular preprocessing improved robustness and generalization.

---


---

## Code & Artifacts
- Code files and notebooks  
- Additional results: Feature importance plots, TF-IDF vocabulary stats, and text-length histograms are included in logs  

---

## Tech Stack
- **Language:** Python  
- **ML / NLP:** TF-IDF, LightGBM, feature engineering  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Evaluation Metrics:** SMAPE, MAE  

---

## Conclusion
A scalable hybrid ML model that efficiently integrates textual semantics and structured features for product price prediction. The approach demonstrates the importance of preprocessing, unit normalization, and modular feature pipelines for robust and generalizable results.

## Usage 
run the ipynb file provided

