# 🚀 Spaceship Titanic — Kaggle Competition

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat&logo=kaggle&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-80.17%25-2ecc71?style=flat)
![Rank](https://img.shields.io/badge/Leaderboard-Top%20820-gold?style=flat)

> **Binary classification challenge** — Predict which passengers aboard the Spaceship Titanic were transported to an alternate dimension after a spacetime anomaly collision.

---

## 🏆 Result

| Metric | Value |
|--------|-------|
| 🎯 Public Leaderboard Score | **0.80173** |
| 🏅 Leaderboard Rank | **#820** |
| 📦 Submissions | 1 (First Entry) |
| 📊 Evaluation Metric | Classification Accuracy |

---

## 📖 Problem Statement

Set in the year **2912**, the luxury interstellar liner *Spaceship Titanic* collided with a hidden spacetime anomaly. Nearly half of the 13,000 passengers were transported to an alternate dimension.

Using personal records recovered from the ship's damaged computer system, the task is to **predict which passengers were transported** — a binary classification problem.

---

## 📁 Repository Structure

```
spaceship-titanic/
├── train.csv                          # Labelled training data (8,693 rows)
├── test.csv                           # Unlabelled test data (4,277 rows)
├── sample_submission.csv              # Kaggle submission template
├── submission_final.csv               # Final model predictions
├── Spaceship_Titanic.ipynb            # Full analysis & modelling notebook
└── README.md                          # This file
```

---

## 📊 Dataset

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 8,693 | Training data with target column `Transported` |
| `test.csv` | 4,277 | Test data for generating predictions |
| `sample_submission.csv` | 4,277 | Required submission format |
| `submission_final.csv` | 4,277 | Our final predictions |

### Key Features

| Feature | Description |
|---------|-------------|
| `PassengerId` | Unique ID in format `GGGG_PP` (group & person) |
| `HomePlanet` | Departure planet (Earth, Europa, Mars) |
| `CryoSleep` | Whether passenger was in suspended animation |
| `Cabin` | Cabin number: `Deck/Num/Side` |
| `Destination` | Planet of disembarkation |
| `Age` | Passenger age |
| `VIP` | Whether special VIP service was purchased |
| `RoomService` / `FoodCourt` / `ShoppingMall` / `Spa` / `VRDeck` | Luxury amenity billing |
| `Transported` | 🎯 **Target** — Was the passenger transported? (True/False) |

---

## 🔬 Methodology

### Step 1 — Data Loading & Combining
Loaded `train.csv` and `test.csv` separately, added an `is_train` flag, then concatenated into a single DataFrame to ensure consistent preprocessing across both splits.

### Step 2 — Exploratory Data Analysis (EDA)
- Inspected shape, dtypes, and head
- Counted null values per column
- Identified and removed duplicate rows

### Step 3 — Feature Engineering

| New Feature | Derived From | Method |
|-------------|-------------|--------|
| `Deck`, `Num`, `Side` | `Cabin` | Split on `/` |
| `TotalSpent` | 5 amenity columns | Row-wise sum (NaN → 0) |
| `Group` | `PassengerId` | First 4 characters |

### Step 4 — Missing Value Imputation
- **Categorical / Boolean** columns → filled with **mode**
- **Age** (numerical) → filled with **median**
- Booleans (`CryoSleep`, `VIP`) → cast to `int` (0/1)

### Step 5 — Encoding & Feature Selection
- Applied `pd.get_dummies()` to: `HomePlanet`, `Destination`, `Deck`, `Side`
- Dropped high-cardinality ID columns: `PassengerId`, `Cabin`, `Name`, `Group`, `Num`

### Step 6 — Model Training with Cross-Validation

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"Mean CV Accuracy: {cv_results.mean():.4f}")
print(f"Std Dev:          {cv_results.std():.4f}")

model.fit(X, y)  # Final fit on full training data
```

### Step 7 — Visualisation
- `HomePlanet` vs `Transported` — count plot
- `CryoSleep` vs `Transported` — count plot
- `Destination` vs `Transported` — count plot
- Age distribution split by `Transported` — KDE plot
- Full numeric correlation matrix — heatmap (coolwarm)

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, preprocessing |
| `NumPy` | Numerical operations |
| `scikit-learn` | Model training, cross-validation, scoring |
| `Matplotlib` | Chart rendering |
| `seaborn` | Statistical visualisation |
| `Jupyter Notebook` | Interactive development environment |

---

## ⚡ Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/your-username/spaceship-titanic.git
cd spaceship-titanic
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

**3. Download the data**

Head to the [Kaggle competition page](https://www.kaggle.com/competitions/spaceship-titanic/data) and download `train.csv` and `test.csv` into the project directory.

**4. Run the notebook**
```bash
jupyter notebook Spaceship_Titanic.ipynb
```

Run all cells in order. The final predictions will be saved to `submission_final.csv`.

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with `GridSearchCV` or **Optuna**
- [ ] Ensemble stacking (XGBoost + LightGBM + GBM)
- [ ] Advanced features — group size, spending ratios, age bins
- [ ] SHAP values for model interpretability
- [ ] Target encoding for high-cardinality features

---

## 👤 Author

**Gururaj Krishna Sharma**
- 🏅 Kaggle: [kaggle.com/gururajkrishnasharma](https://www.kaggle.com/gururajkrishnasharma)
- 🎯 Competition: Spaceship Titanic — First Entry · Rank **#820** · Score **0.80173**

---

<p align="center">
  <i>Built with Python & scikit-learn · Kaggle Spaceship Titanic Competition 2025</i>
</p>
