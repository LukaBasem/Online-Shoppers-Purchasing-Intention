# 🛒 Online Shoppers Purchase Intention Prediction

**NTI – Machine Learning for Data Analysis | Final Project**

Predict whether an online-shopping session will end in a **purchase**, using the visitor's behaviour during the session. The best model is deployed as a **Gradio** web app.

🔗 **Live Demo:** `<paste your Hugging Face Space link here>`
💻 **GitHub:** `<paste your repository link here>`

---

## 1. Problem Statement

An online store wants to know, while a visitor is browsing, **whether the session will end in a purchase**. With this information the store can show offers only to visitors who are likely to buy but have not decided yet, and avoid wasting marketing effort on the rest.

- **Task:** binary classification
- **Target:** `Revenue` (1 = purchase, 0 = no purchase)
- **Main challenge:** imbalanced data (~15.5% buyers) → accuracy alone is misleading, so the main metric is **F1-score** (with Recall, ROC-AUC and PR-AUC)

## 2. Dataset

`online_shoppers_intention.csv` – **12,330 sessions**, 17 features + target, no missing values, 125 duplicate rows (removed).

| Group | Features |
|---|---|
| Pages & time | `Administrative`, `Informational`, `ProductRelated` and their `_Duration` columns |
| Behaviour metrics | `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay` |
| Context | `Month`, `VisitorType`, `Weekend`, `OperatingSystems`, `Browser`, `Region`, `TrafficType` |
| **Target** | `Revenue` |

## 3. Project Structure

```
Final_Project/
├── dataset/
│   └── online_shoppers_intention.csv
├── notebooks/
│   └── final_project.ipynb        # full analysis: EDA → models → tuning → evaluation
├── models/
│   └── best_model.joblib          # saved pipeline (preprocessing + Random Forest)
├── results/                       # all charts (.png) and result tables (.csv)
├── app.py                         # Gradio app
├── requirements.txt               # libraries needed to run the app
└── README.md
```

## 4. What Was Done

| Step | Details |
|---|---|
| **EDA** | Target distribution, feature distributions, purchase rate by month / visitor type / weekend, buyers vs non-buyers, correlation heatmap, chi-square tests |
| **Cleaning** | No missing values; removed 125 duplicates; outliers checked with the IQR rule and **kept** (they are real behaviour and the best models are tree-based) |
| **Feature engineering** | `TotalPages`, `TotalDuration`, `AvgTimePerPage` |
| **Preprocessing** | `ColumnTransformer` inside a `Pipeline`: `StandardScaler` for numeric features, `OneHotEncoder` for categorical features |
| **Data leakage** | Stratified 80/20 split **before** any fitting; the scaler/encoder are learned on training folds only; the test set is used only once, at the end |
| **Models (11)** | Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, Gradient Boosting, AdaBoost, Extra Trees, XGBoost, LightGBM |
| **Comparison** | Stratified 5-fold cross-validation on the training set, ranked by F1 |
| **Tuning** | `GridSearchCV` (scoring = F1, 5-fold) on the top 3 models, comparison before vs after |
| **Final evaluation** | Test set: Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report, ROC Curve + AUC, Precision-Recall Curve + AP |
| **Deployment** | Gradio app that loads the saved pipeline and predicts from raw input |

## 5. Results

### Model comparison (5-fold CV on the training set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|:--|--:|--:|--:|--:|--:|
| **Random Forest** | 0.896 | 0.655 | 0.718 | **0.684** | 0.922 |
| LightGBM | 0.882 | 0.592 | 0.778 | 0.672 | 0.928 |
| XGBoost | 0.888 | 0.629 | 0.698 | 0.662 | 0.919 |
| Gradient Boosting | 0.901 | 0.723 | 0.594 | 0.652 | 0.929 |
| SVM | 0.868 | 0.560 | 0.746 | 0.639 | 0.901 |
| AdaBoost | 0.890 | 0.675 | 0.568 | 0.617 | 0.913 |
| Logistic Regression | 0.847 | 0.508 | 0.754 | 0.607 | 0.899 |
| Decision Tree | 0.861 | 0.557 | 0.550 | 0.553 | 0.734 |
| KNN | 0.871 | 0.655 | 0.367 | 0.470 | 0.779 |
| Extra Trees | 0.876 | 0.772 | 0.289 | 0.420 | 0.889 |
| Naive Bayes | 0.276 | 0.176 | 0.981 | 0.298 | 0.702 |

### GridSearchCV (top 3 models)

| Model | CV F1 before | CV F1 after |
|:--|--:|--:|
| **Random Forest** | 0.6841 | **0.6841** |
| LightGBM | 0.6725 | 0.6751 |
| XGBoost | 0.6615 | 0.6685 |

Random Forest stays the best after tuning; its default parameters were already the best combination in the grid, so tuning did not change it. This is a real result, not an error.

### Final model on the test set (Random Forest)

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) |
|--:|--:|--:|--:|--:|--:|
| 0.893 | 0.647 | 0.702 | 0.673 | 0.925 | 0.718 |

Confusion matrix (2,441 sessions): 1,913 true negatives · 146 false positives · 114 false negatives · **268 true positives**.

### Key insights

- `PageValues` is by far the most important feature, followed by `ExitRates` and the time/pages features.
- November has the highest purchase rate (25.4%); February the lowest (1.6%).
- Returning visitors are ~86% of the traffic but convert less (13.9%) than new visitors (24.9%).

### Limitations

- `PageValues` is calculated from pages that led to a transaction, so it may partly "know" the purchase. In a live system, make sure it is available *before* the purchase happens.
- The split is random, not time-based; only ~12k sessions; January and April are missing; the default 0.5 threshold was used.

## 6. How to Run

**1) Install the libraries**

```bash
pip install -r requirements.txt
```

**2) Run the Gradio app**

```bash
python app.py
```

Then open the local link shown in the terminal (usually `http://127.0.0.1:7860`).

**3) (Optional) Re-run the notebook**

```bash
pip install matplotlib seaborn scipy xgboost lightgbm jupyter
jupyter notebook notebooks/final_project.ipynb
```

Running the notebook from top to bottom recreates `models/best_model.joblib` and everything inside `results/`.

> The code was tested with Python 3.12 and the library versions in `requirements.txt`. If you use different versions, re-run the notebook once so the saved model matches your scikit-learn version.

## 7. Deployment on Hugging Face Spaces (Live Link)

1. Create a free account on [huggingface.co](https://huggingface.co) → **New Space** → SDK: **Gradio**.
2. Upload these files, keeping the same folder structure: `app.py`, `requirements.txt`, and `models/best_model.joblib`.
3. If the build asks for a Python version, add `python_version: 3.12` to the header (between the `---` lines) at the top of the Space's `README.md`.
4. Wait for the build to finish, test the app, and copy the Space link into the **Live Demo** line at the top of this file.

## 8. Author

**Luka Basem Wagdy** – Computer Science (AI major), NTI Machine Learning for Data Analysis.
