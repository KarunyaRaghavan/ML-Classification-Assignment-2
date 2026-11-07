# Bank Marketing Classification – Machine Learning Assignment 2

## a. Problem statement
Build and evaluate five supervised classification models to predict whether a bank customer subscribes to a term deposit (`y`: yes/no). The project also provides a Streamlit interface to evaluate uploaded labelled test data using saved models.

## b. Dataset description
- Source: UCI Machine Learning Repository – Bank Marketing
- Dataset: `bank-full.csv`
- Instances: 45,211
- Input features: 16
- Target: `y` (yes/no), binary classification
- Dataset page: https://archive.ics.uci.edu/dataset/222/bank+marketing
- The training script downloads the original UCI ZIP and reads `bank-full.csv`.

## c. GitHub Repository Link
PASTE YOUR PUBLIC GITHUB REPOSITORY URL HERE.

## d. Models used and evaluation metrics
Run `python train_models.py`. Copy the actual values from the generated `metrics.csv` into this table. Do not invent or estimate metric values.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations on model performance
Write observations only after reviewing your actual `metrics.csv`. Discuss the trade-off between accuracy and recall, the AUC ranking, and whether the best MCC/F1 model is also the overall winner.

Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.8457	0.9079	0.4182	0.8147	0.5527	0.5092
Decision Tree	0.8397	0.8846	0.4101	0.8450	0.5522	0.5139
KNN	0.8996	0.8513	0.6298	0.3440	0.4450	0.4169
Naive Bayes	0.8548	0.8101	0.4059	0.5198	0.4559	0.3774
SVM	0.8456	0.9256	0.4225	0.8715	0.5691	0.5361

Overall winner: SVM, based on its highest AUC, Recall, F1 and MCC.

## Project files
- `train_models.py`: downloads data, preprocesses, trains and evaluates all five models, saves models and test data.
- `app.py`: Streamlit application for test CSV upload, model selection, metrics, confusion matrix and classification report.
- `test_data.csv`: held-out test data including the target column.
- `model/`: saved trained model pipelines.
- `metrics.csv`: measured model comparison results.

## How to run locally
1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Train and save models: `python train_models.py`
4. Start the app: `streamlit run app.py`
5. Upload the generated `test_data.csv` in the app.

## Methodology
The dataset is split into 80% training and 20% testing using stratified sampling (`random_state=42`). Numerical variables are standardized and categorical variables are one-hot encoded. The same preprocessing and train/test split are used for all five classifiers. The uploaded test data is evaluated by loading saved pipelines; the app does not retrain models.

## Streamlit deployment
Push the complete project to GitHub, including the `model/` directory and `test_data.csv`. In Streamlit Community Cloud, create a new app, select the repository and `main` branch, choose `app.py`, and deploy. Paste the live app URL here: PASTE STREAMLIT APP URL.
