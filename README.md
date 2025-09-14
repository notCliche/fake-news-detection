# Fake News Classification Project

## Introduction
In an age of rapid information dissemination, the ability to automatically discern credible news from misinformation is a critical challenge. This project tackles that challenge by building and comparing several classification models. The primary dataset is sourced from Kaggle and contains a collection of news articles with various attributes, including a label indicating their authenticity.

## Project Overview
This project aims to classify news articles as **"Real"** or **"Fake"** using machine learning and natural language processing (NLP). We explore three distinct modeling techniques, progressively increasing in complexity and performance:

1. **Random Forest Classifier** → A strong baseline with traditional feature engineering.  
2. **LightGBM (LGBM) Classifier** → A high-performance gradient boosting model designed for speed and accuracy.  
3. **DistilBERT Transformer** → A state-of-the-art deep learning model leveraging contextual word embeddings for deeper text understanding.  

Each approach is contained in its own notebook, allowing direct comparison of their effectiveness.

## Dataset
The dataset used is [`news_articles.csv`](news_articles.csv), sourced from [Kaggle](https://www.kaggle.com/datasets/ruchi798/source-based-news-classification).

We focus primarily on the following columns:

- **title** → The headline of the article (often sensational in fake news).  
- **text** → The main body of the article.  
- **site_url** → Source website (domain credibility can be useful).  
- **hasImage** → Binary indicator of whether an article has an image.  
- **label** → Target variable (`Real` or `Fake`).  

## Methodology & Modeling Approaches

### Approach 1: Random Forest with Feature Engineering
**File:** [`fakenewsrf.ipynb`](fakenewsrf.ipynb)  

This baseline model combines classic NLP with engineered features.  

- **Text Vectorization** → TF-IDF applied to `title` and `text`.  
- **Feature Engineering**:
  - Text length & word counts.  
  - Source domain credibility (from `site_url`).  
  - Stylistic features (all-caps words, punctuation counts, etc.).  
- **Model** → `RandomForestClassifier` trained on combined TF-IDF + features.  

### Approach 2: LightGBM Classifier
**File:** [`fakenewslgbm.ipynb`](fakenewslgbm.ipynb)  

A more powerful model replacing Random Forest with LightGBM.  

- **Same feature engineering & TF-IDF** as above for fair comparison.  
- **Model** → `lgb.LGBMClassifier`, leveraging gradient boosting.  
  - Builds trees sequentially to correct errors.  
  - Provides improved accuracy and efficiency.  

### Approach 3: DistilBERT Transformer
**File:** [`fakenewstf.ipynb`](fakenewstf.ipynb)  

The most advanced approach, using a pre-trained Transformer model.  

- **No manual feature engineering** needed.  
- **Tokenization** → `DistilBertTokenizerFast` for sub-word encoding, padding, and attention masks.
- **Model** → Fine-tuned `DistilBERT` for classification.
  - Learns contextual meaning of words.
  - Expected to yield the **highest accuracy**.
  - Requires GPU for efficient training.

## How to Run

### Prerequisites
Install dependencies:
```
pip install pandas scikit-learn matplotlib seaborn lightgbm
pip install transformers torch datasets
````
### Instructions

1. Place **`news_articles.csv`** in the project directory.
2. Choose a model:

   * Run `fakenewsrf.ipynb` → Random Forest.
   * Run `fakenewslgbm.ipynb` → LightGBM.
   * Run `fakenewstf.ipynb` → Transformer (DistilBERT).
3. View results:

   * Accuracy, precision, recall, F1-score.
   * Confusion matrix plots.
   * Predictions saved in CSV:

     * `prediction_results.csv`
     * `prediction_results_lgbm.csv`
     * `prediction_results_transformer.csv`

## Results & Conclusion

* **Random Forest** → Solid baseline with engineered features.
* **LightGBM** → Faster and more accurate than Random Forest.
* **DistilBERT** → Best results due to contextual language understanding, but requires more compute.

## Author
Om Prakash Behera
