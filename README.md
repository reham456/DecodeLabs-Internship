# Titanic Survival Prediction — Project README

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Data Preprocessing](#data-preprocessing)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Feature Engineering](#feature-engineering)
7. [Predictive Model](#predictive-model)
8. [Model Evaluation](#model-evaluation)
9. [Results & Visualizations](#results--visualizations)
10. [Example Predictions](#example-predictions)
11. [Why Each Method Was Chosen](#why-each-method-was-chosen)
12. [Limitations & Honest Accuracy Ceiling](#limitations--honest-accuracy-ceiling)
13. [Dependencies](#dependencies)

---

## Project Overview

This project builds a machine learning pipeline to predict whether a passenger aboard the RMS Titanic survived or perished, based on demographic and travel-related features. The pipeline covers the full data science workflow: cleaning raw data, engineering meaningful features, training a high-performance ensemble model, and evaluating it rigorously using cross-validation.

**Target variable:** `Survived` (1 = Survived, 0 = Did Not Survive)  
**Best achievable accuracy on this dataset:** ~83–85% (an established ceiling due to noise and missing data inherent to the dataset)

---

## Dataset

**Source:** `Titanic-Dataset.csv`  
**Size:** 891 rows × 12 columns (raw)

| Column | Description |
|---|---|
| `PassengerId` | Unique passenger identifier |
| `Survived` | Survival label (0 = No, 1 = Yes) |
| `Pclass` | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name` | Full passenger name |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare paid |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## Project Structure

```
Task5.ipynb              ← Main notebook (preprocessing + model)
Titanic-Dataset.csv      ← Raw dataset
task5_model.png          ← Output visualization (confusion matrix, feature importance, CV scores)
README.md                ← This file
```

---

## Data Preprocessing

The raw dataset was cleaned in a series of deliberate steps before any modelling was performed.

### 1. Handling Missing Values

| Column | Missing % | Strategy | Reason |
|---|---|---|---|
| `Cabin` | ~77% | **Dropped entirely** | Over three-quarters of values are missing — too sparse to impute meaningfully or extract reliable patterns from |
| `Age` | ~20% | **Filled with median** | Median is robust to outliers (e.g. very old passengers skew the mean); preserves the central tendency without distortion |
| `Embarked` | ~0.2% (2 rows) | **Filled with mode** | Only 2 values missing; the most frequent port (Southampton) is the safe and minimal-impact choice |

### 2. Removing Duplicates

Duplicate rows were identified and removed using `df.drop_duplicates()`. This prevents the model from learning the same record multiple times, which would artificially inflate its confidence on repeated patterns.

### 3. Label Encoding Categorical Variables

Machine learning models require numerical input. Three categorical columns were converted:

| Column | Encoding | Example |
|---|---|---|
| `Sex` | Label Encoding | female → 1, male → 0 |
| `Embarked` | Label Encoding | C → 0, Q → 1, S → 2 |
| `Title` | Label Encoding | Master → 0, Miss → 1, Mr → 2, Mrs → 3, Rare → 4 |

Label encoding (rather than one-hot encoding) was chosen because tree-based models — used later — handle ordinal integer labels natively and do not require dummy variables.

### 4. Dropping Irrelevant Columns

`PassengerId`, `Name`, and `Ticket` were dropped because:
- `PassengerId` is an arbitrary row index with no predictive signal.
- `Name` as raw text is not usable directly — the useful information (title) was extracted into a separate feature first.
- `Ticket` contains alphanumeric codes with no consistent structure that could be reliably mapped to survival.

### 5. Outlier Analysis (IQR Method)

An outlier check was performed on `Age`, `Fare`, and `FamilySize` using the Interquartile Range (IQR) method. Values below `Q1 − 1.5×IQR` or above `Q3 + 1.5×IQR` were flagged. This step was analytical rather than corrective — outliers were not removed, because extreme fares (e.g. wealthy 1st-class passengers) and unusual family sizes are genuine and informative signals, not data errors.

---

## Exploratory Data Analysis

Before building the model, the data was analysed along four dimensions to understand the survival patterns:

- **By Gender:** Female passengers had a dramatically higher survival rate than males, reflecting the "women and children first" evacuation policy.
- **By Passenger Class:** 1st class passengers survived at much higher rates than 3rd class, reflecting proximity to lifeboats and preferential access.
- **By Title:** Titles like `Mrs` and `Miss` showed high survival rates; `Mr` showed low survival, and `Rare` titles (Doctors, Colonels, etc.) varied.
- **By Age Group:** Children (under 12) had notably higher survival rates; seniors (over 60) had the lowest. Young adults had middling rates.

These patterns directly motivated the feature engineering choices in the next step.

---

## Feature Engineering

After preprocessing, seven additional features were engineered to give the model richer, more discriminating signals.

### Features from Preprocessing Stage

| Feature | Construction | Why |
|---|---|---|
| `Title` | Extracted via regex from `Name` | Titles encode gender, social class, and age simultaneously in a single variable. `Mr` vs `Mrs` vs `Master` captures survival-relevant identity more precisely than gender alone |
| `FamilySize` | `SibSp + Parch + 1` | Combines two correlated columns into a single measure. Passengers in mid-sized families survived better than those alone or in very large groups |
| `IsAlone` | 1 if `FamilySize == 1` | Binary flag highlighting the highest-risk group — solo travellers had lower survival rates |

### Features from Model Stage

| Feature | Construction | Why |
|---|---|---|
| `AgeBand` | Age binned into [Child, Teen, Young Adult, Adult, Senior] | Converts continuous age into survival-relevant brackets; reduces noise from exact age values |
| `FareBand` | Fare split into 4 equal-frequency quartiles | Smooths extreme fare values and groups passengers by relative wealth tier |
| `Pclass_Sex` | `Pclass × Sex_enc` | Captures the interaction between class and gender — 1st class women had the highest survival rate of any group. Neither feature alone captures this joint effect |
| `Age_Pclass` | `Age × Pclass` | Captures that age affected survival differently depending on class — elderly 3rd class passengers fared far worse than elderly 1st class passengers |
| `IsChild` | 1 if `Age < 12` | Hard flag for children, who were prioritised during evacuation regardless of class |
| `IsElderly` | 1 if `Age > 60` | Hard flag for elderly passengers who faced mobility constraints |
| `IsMother` | 1 if female, adult, `Parch > 0`, title is Mrs | Mothers with children had strong survival priority; this composite flag captures that specific pattern |

---

## Predictive Model

### Architecture — Soft Voting Ensemble

Rather than using a single model, three complementary models are combined into a **soft-voting ensemble**. Each model outputs a survival probability, and the final prediction is the weighted average of those probabilities.

```
Final Probability = (3 × XGBoost_prob + 2 × RandomForest_prob + 2 × GradientBoosting_prob) / 7
```

#### Why an Ensemble?
Each model has different strengths and failure modes. Combining them reduces variance — when one model is wrong, the others compensate. Soft voting (probability averaging) is preferred over hard voting (majority vote) because it uses more information from each model.

---

### Model 1 — XGBoost (Weight: 3)

```python
XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.0
)
```

**Why XGBoost?**
XGBoost (Extreme Gradient Boosting) builds trees sequentially — each tree corrects the residual errors of the previous one. It is the strongest individual model on structured tabular data and consistently tops Kaggle competitions on datasets like Titanic. It is given the highest weight (3) because it delivers the best individual accuracy.

**Why these hyperparameters?**
- `max_depth=4` — shallow trees prevent overfitting on a small 891-row dataset
- `learning_rate=0.05` — slow, careful learning with many trees is more accurate than fast learning with few
- `subsample=0.8` + `colsample_bytree=0.8` — random sampling of rows and columns per tree adds diversity and reduces overfitting
- `reg_alpha=0.1` + `reg_lambda=1.0` — L1 and L2 regularisation penalise complexity to keep the model generalising

---

### Model 2 — Random Forest (Weight: 2)

```python
RandomForestClassifier(
    n_estimators=300, max_depth=6,
    min_samples_leaf=3, max_features="sqrt"
)
```

**Why Random Forest?**
Random Forest builds many independent decision trees, each trained on a random bootstrap sample with a random subset of features. Because each tree is trained independently, Random Forest errors are uncorrelated with XGBoost's errors — this diversity is what makes the ensemble stronger than either model alone.

**Why these hyperparameters?**
- `max_depth=6` — allows each tree to learn meaningful patterns without memorising noise
- `min_samples_leaf=3` — requires at least 3 samples at each leaf, smoothing out individual outliers
- `max_features="sqrt"` — standard best practice: each split considers √n features, enforcing diversity across trees

---

### Model 3 — Gradient Boosting (Weight: 2)

```python
GradientBoostingClassifier(
    n_estimators=300, max_depth=4,
    learning_rate=0.05, subsample=0.8
)
```

**Why Gradient Boosting?**
Scikit-learn's Gradient Boosting is an older but stable sequential boosting algorithm. Its implementation differs from XGBoost (different regularisation, no column subsampling by default), so it adds a third perspective to the ensemble that is correlated with but not identical to XGBoost — contributing useful diversity.

---

### Why Soft Voting Over Hard Voting?

Hard voting takes the majority class prediction. Soft voting averages the probabilities. Soft voting is better because:
- A model that is 95% confident one way should outweigh a model that is 51% confident the other way — hard voting cannot capture this
- Probability calibration is more nuanced and produces more accurate final decisions

---

## Model Evaluation

### Train/Test Split

- **80% training** / **20% test** split, stratified by survival label
- Stratification ensures both splits have the same proportion of survivors and non-survivors, preventing an unlucky split from skewing results

### 10-Fold Stratified Cross-Validation

The test set accuracy from a single split is unreliable — it depends partly on which 20% of rows were held out. To get a robust, variance-aware accuracy estimate:

- The full dataset is split into 10 folds
- The model is trained on 9 folds and tested on 1, repeated 10 times
- The mean ± standard deviation across all 10 runs is reported

This is the most honest accuracy metric and is the number that should be cited when reporting model performance.

---

## Results & Visualizations

Three plots are saved to `task5_model.png`:

**1. Confusion Matrix**  
Shows the breakdown of correct and incorrect predictions:
- True Negatives (top-left): correctly predicted did not survive
- True Positives (bottom-right): correctly predicted survived
- False Positives (top-right): predicted survived but actually did not
- False Negatives (bottom-left): predicted did not survive but actually survived

**2. Top 15 Feature Importances (XGBoost)**  
Ranks features by how much they reduce prediction error inside XGBoost. Features like `Sex_enc`, `Pclass_Sex`, `Title_enc`, and `Fare` typically dominate, confirming the historical narrative that gender, class, and wealth were the primary survival determinants.

**3. 10-Fold Cross-Validation Scores**  
Bar chart of accuracy across all 10 folds with a dashed mean line, giving a visual sense of model consistency. High consistency (low spread across folds) means the model is stable and not overfitting to any particular subset of the data.

---

## Example Predictions

Three hand-crafted passenger profiles demonstrate the model's real-world behaviour:

| Passenger Profile | Prediction | Survival Probability | Reasoning |
|---|---|---|---|
| Mrs, 1st class, age 35, with spouse | ✅ Survived | High | Female + high class = top evacuation priority |
| Mr, 3rd class, age 22, alone | ❌ Did Not Survive | Low | Male + lowest class + no family = lowest priority |
| Child, 2nd class, age 8, with parents | ✅ Survived | High | Children prioritised regardless of class; with parents adds stability |

---

## Why Each Method Was Chosen

| Decision | Choice Made | Why Not the Alternative |
|---|---|---|
| Missing `Cabin` | Dropped | 77% missing — any imputation would be fabricating data for most rows |
| Missing `Age` | Median imputation | Mean is distorted by outlier ages; median is more representative of the typical passenger |
| Missing `Embarked` | Mode imputation | Only 2 rows; any sophisticated method would be overkill |
| Encoding | Label encoding | One-hot encoding is unnecessary for tree-based models and inflates dimensionality |
| Feature interactions | `Pclass_Sex`, `Age_Pclass` | Linear models need explicit interactions; tree models benefit too since they can split on derived values directly |
| Model type | Ensemble (XGBoost + RF + GB) | Single models have higher variance; ensembles consistently outperform on structured data |
| Voting type | Soft voting | Uses probability information rather than discarding confidence levels |
| Validation | 10-fold CV | Single test split is luck-dependent; 10-fold gives a reliable, variance-aware estimate |
| XGBoost weight | 3 vs 2 for others | XGBoost delivers the highest individual accuracy and should influence the final vote more |
| Outlier strategy | Analyse, do not remove | Extreme fares and family sizes are genuine signals, not data entry errors |

---

## Limitations & Honest Accuracy Ceiling

The Titanic dataset has a well-documented accuracy ceiling of approximately **83–85%** for legitimate models. This is not a limitation of the modelling approach — it reflects irreducible noise in the data:

- Survival had a significant random component (proximity to a lifeboat, timing, chaos on deck)
- ~20% of `Age` values were missing and had to be imputed
- `Cabin` — a strong proxy for deck location and lifeboat access — was mostly missing and had to be discarded
- The dataset contains only 891 rows, limiting the statistical power of any model

Any model reporting accuracy significantly above 85% on this dataset should be treated with suspicion — it almost certainly indicates overfitting or data leakage.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
