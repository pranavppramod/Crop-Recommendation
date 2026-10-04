# Crop Recommendation System

![Preview Banner](./docs/banner.png)

A machine learning system that recommends a suitable crop based on soil and environmental conditions using a tuned Random Forest classifier.

The project covers the complete machine learning workflow — from exploratory data analysis and model comparison to hyperparameter tuning, error analysis, explainability, model serialization, and a Streamlit application.

## 📌 Project Overview

Choosing a suitable crop depends on multiple soil and environmental factors such as nitrogen, phosphorus, potassium, temperature, humidity, soil pH, and rainfall.

This project uses these seven parameters to classify an input into one of **22 crop categories**.

The final model is a tuned **Random Forest Classifier** trained on 2,200 samples.

### Supported Crops

Apple, Banana, Blackgram, Chickpea, Coconut, Coffee, Cotton, Grapes, Jute, Kidney Beans, Lentil, Maize, Mango, Moth Beans, Mung Bean, Muskmelon, Orange, Papaya, Pigeon Peas, Pomegranate, Rice, Watermelon.

---

## 📊 Dataset

The dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/discussion/232252). It contains **2,200 samples** with seven numerical input features and one categorical target.

| Feature | Description |
|---|---|
| `N` | Nitrogen content |
| `P` | Phosphorus content |
| `K` | Potassium content |
| `temperature` | Temperature |
| `humidity` | Relative humidity |
| `ph` | Soil pH |
| `rainfall` | Rainfall |
| `label` | Recommended crop |

The dataset is perfectly balanced, with **100 samples per crop class**.

---

## 🔎 Exploratory Data Analysis

The EDA phase investigated:

- Dataset structure and data types
- Missing values and duplicate records
- Class distribution
- Descriptive statistics
- Outlier detection using the IQR method
- Pearson correlation analysis
- Crop-wise feature distributions
- ANOVA-based feature-to-target analysis
- PCA-based dimensionality visualization

### Key Findings

- No missing values or duplicate records were found.
- All 22 crop classes contain exactly 100 samples.
- Potassium (`K`) showed the strongest univariate class separation according to ANOVA.
- `P` and `K` showed the strongest global feature-feature correlation (`r ≈ 0.736`).
- Several global IQR outliers were legitimate crop-specific patterns, so they were retained rather than removed.
- PCA showed meaningful structure in the feature space while also revealing overlap between some crop classes.

---

## 🤖 Model Development

Four classification algorithms were compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

The models were evaluated using stratified 5-fold cross-validation on the training set.

### Baseline Cross-Validation Results

| Model | Mean CV Accuracy | Std. Dev. |
|---|---:|---:|
| Random Forest | 99.38% | 0.61% |
| Gradient Boosting | 98.69% | 0.34% |
| Decision Tree | 98.52% | 0.68% |
| Logistic Regression | 96.82% | 0.66% |

Random Forest achieved the highest mean validation accuracy and was selected for further tuning.

---

## ⚙️ Hyperparameter Tuning

`GridSearchCV` was used with stratified 5-fold cross-validation.

Final Random Forest configuration:

```text
n_estimators = 200
max_depth = None
min_samples_split = 5
min_samples_leaf = 1
criterion = gini
random_state = 42
```

### Tuned Cross-Validation Performance

**Mean CV Accuracy:** 99.60%  
**CV Standard Deviation:** 0.50%

The tuned model improved both mean validation accuracy and fold-to-fold consistency compared with the baseline Random Forest.

---

## 📈 Final Test Performance

The final model was evaluated once on the untouched 20% test set.

| Metric | Score |
|---|---:|
| Accuracy | **99.55%** |
| Precision | **99.57%** |
| Recall | **99.55%** |
| F1 Score | **99.55%** |
| Correct Predictions | **438 / 440** |
| Incorrect Predictions | **2 / 440** |

The two misclassified samples were:

- Blackgram → Maize
- Rice → Jute

These errors occurred in regions where the feature combinations of the respective crop classes overlap, rather than indicating a broad systematic failure.

---

## 🧠 Model Explainability

Several complementary techniques were used to understand the model.

### Feature Importance

Random Forest impurity-based importance ranked the features approximately as:

1. Rainfall
2. Humidity
3. K
4. P
5. N
6. Temperature
7. pH

### Permutation Importance

Permutation importance on the test set ranked:

1. Humidity
2. N
3. Rainfall
4. K
5. P
6. Temperature
7. pH

The differences between these rankings demonstrate why feature importance should not be interpreted using a single method.

### SHAP

SHAP was used during analysis for global and class-specific explainability.

Global SHAP analysis showed that humidity had the strongest average contribution, followed by K, N, rainfall, and P, while temperature and pH had smaller contributions.

For the rice class, rainfall had the largest average SHAP impact, followed by N, humidity, K, P, temperature, and pH.

SHAP was used as an analysis tool and was not included for prediction explainability in the deployed application.

---

## 🧪 Feature Ablation

The contribution of the lower-ranked features was tested by removing them and repeating cross-validation.

| Feature Set | Mean CV Accuracy |
|---|---:|
| All Features | **99.60%** |
| Without Temperature | 99.26% |
| Without pH | 99.32% |
| Without Temperature & pH | 99.03% |

Although temperature and pH had relatively low feature importance, removing either one reduced validation performance. Therefore, all seven features were retained.

---

## 💾 Model Serialization

The trained Random Forest model was serialized using `joblib`:

```text
models/crop_recommendation_model.pkl
```

The serialized model was loaded again and tested against the original test predictions.

**Serialization integrity check: PASSED**

The loaded model produced predictions identical to those generated before serialization.

---

## 🖥️ Streamlit Application

The project includes a Streamlit interface where users can enter:

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- Soil pH
- Rainfall

The application returns:

- Recommended crop
- Model prediction confidence
- Entered input values
- Top alternative predictions

### Run Locally

Clone the repository:

```bash
git clone https://github.com/pranavppramod/Crop-Recommendation
cd Crop-Recommendation
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
Crop-Recommendation/
│
├── config/
│   └── metadata.json
│
├── dataset/
│   └── original.csv
│
├── models/
│   └── crop_recommendation_model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

### File Overview

- `config/metadata.json` — project metadata, dataset information, model parameters, performance metrics, feature ranges, and deployment configuration.
- `dataset/original.csv` — dataset used for model development.
- `models/crop_recommendation_model.pkl` — serialized tuned Random Forest model.
- `app.py` — Streamlit application.
- `requirements.txt` — deployment dependencies.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- SHAP

SHAP was used during model analysis and explainability but is not required to run the Streamlit application.

---

## 🔬 Machine Learning Workflow

```text
Dataset
   ↓
Data Inspection
   ↓
Exploratory Data Analysis
   ↓
Outlier & Correlation Analysis
   ↓
ANOVA & PCA
   ↓
Train/Test Split
   ↓
Baseline Model Comparison
   ↓
Cross-Validation
   ↓
Random Forest Hyperparameter Tuning
   ↓
Final Test Evaluation
   ↓
Error Analysis
   ↓
Feature Importance / Permutation Importance / SHAP
   ↓
Feature Ablation
   ↓
Model Serialization
   ↓
Streamlit Application
```

---

## ⚠️ Disclaimer

This application provides machine-learning-based crop recommendations based on patterns learned from the training dataset.

The output should **not** be treated as a substitute for professional agricultural advice, soil testing, or location-specific agronomic assessment.

---

## 👨‍💻 Author

**Pranav Pramod**

B.Tech Computer Science student learning AIML Engineering. This is my third project on Multi Class Classification, made during the rush of semester end.
