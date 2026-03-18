# Hate Speech Detection using NLP & Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLP-NLTK-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## What This Project Is About

Online platforms process hundreds of millions of posts every day. Manually moderating harmful content at that scale is impossible — which is why automated hate speech detection is one of the most practically relevant NLP problems today.

This project builds a complete text classification pipeline to automatically detect and categorise hate speech in tweets, distinguishing between three classes: **hate speech**, **offensive language**, and **neutral content**. The pipeline covers everything from raw text preprocessing through to model training and live prediction on new inputs.

---

## Dataset

| File | Description |
|------|-------------|
| `labeled_data.csv` | ~24,000 tweets labeled by crowdsourced annotators |

**Label mapping:**

| Class | Label |
|-------|-------|
| 0 | Hate Speech |
| 1 | Offensive Language |
| 2 | No Hate or Offensive Language |

**Source:** [Hate Speech and Offensive Language Dataset — Kaggle](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

---

## Pipeline

### 1. Data Loading & Exploration
- Loaded `labeled_data.csv` and mapped integer class labels to readable strings
- Checked for null values, data types, and class distribution
- Isolated the two working columns: `tweet` and `labels`

### 2. Text Preprocessing
Each tweet is passed through a cleaning function that:
- Lowercases all text
- Strips URLs (`http://`, `www.`)
- Removes HTML tags, punctuation, and newline characters
- Removes **English stopwords** via NLTK
- Applies **Snowball Stemmer** — reduces words to their root form (e.g., "running", "runs" → "run")

### 3. Feature Extraction
- **Count Vectorizer (Bag of Words):** converts cleaned tweet text into a sparse numerical matrix where each column represents a unique token in the vocabulary
- `fit_transform` applied on training data; `transform` only on test/inference data to prevent data leakage

### 4. Model Training
- **Decision Tree Classifier** trained on a 67/33 train-test split
- Baseline model to establish performance floor

### 5. Evaluation
- **Confusion matrix** visualised as an annotated seaborn heatmap
- **Accuracy score** via `sklearn.metrics`

### 6. Live Inference
The trained model is tested on unseen sentences:

```python
# Example 1
input = "Let's unite and kill all the people who are protesting against the government"
# Predicted: Hate Speech

# Example 2
input = "she is a bad person"
# Predicted: Offensive Language
```

---

## Results

| Component | Detail |
|-----------|--------|
| Model | Decision Tree Classifier |
| Feature Extraction | Count Vectorizer (Bag of Words) |
| Text Cleaning | Lowercase + URL/HTML removal + stopwords + stemming |
| Train / Test Split | 67% / 33% |
| Evaluation | Confusion Matrix + Accuracy Score |

---

## Tech Stack

```
Python 3 · pandas · numpy · nltk · scikit-learn · matplotlib · seaborn · Jupyter Notebook
```

---

## Project Structure

```
hate-speech-detection/
├── hate_speech_detection.ipynb  # Full notebook
├── labeled_data.csv             # Tweet dataset
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/RIDA-IYENGAR/hate-speech-detection.git
cd hate-speech-detection
pip install pandas numpy matplotlib seaborn scikit-learn nltk jupyter
jupyter notebook hate_speech_detection.ipynb
```

> NLTK stopwords are downloaded automatically on first run via `nltk.download('stopwords')`.

---

## Potential Next Steps

- Test stronger classifiers: Logistic Regression, Random Forest, SVM, or fine-tuned BERT
- Address class imbalance using SMOTE or class-weighted loss
- Replace Count Vectorizer with TF-IDF or word embeddings (Word2Vec, GloVe)
- Add cross-validation for more reliable performance estimates
- Deploy as a REST API or lightweight web app for real-time moderation

---

## About

**Rida Iyengar** · Biotechnology Engineering, PES University  
[GitHub](https://github.com/RIDA-IYENGAR)

