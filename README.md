# Predictive Modeling Using Machine Learning

A beginner-friendly supervised machine learning project that trains models to predict customer churn from a sample dataset.

## Features

- Load and inspect a dataset
- Prepare features and target values
- Split data into training and testing sets
- Train multiple models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Compare accuracy, precision, recall, F1 score, and ROC AUC
- Generate a confusion matrix visualization
- Generate a ROC curve visualization
- Save the best trained model
- Create a summary report

## Project Structure

```text
predictive-modeling-ml-project/
  data/
    customer_churn.csv
  models/
  reports/
    figures/
  src/
    data_loader.py
    train_model.py
    evaluate_model.py
    main.py
  requirements.txt
  README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## Run the Project

```bash
python src/main.py
```

After running, check:

- `models/best_model.joblib`
- `reports/model_report.md`
- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curve.png`

## Dataset

The included dataset is a small sample customer churn dataset. The target column is `churn`, where:

- `0` means the customer did not churn
- `1` means the customer churned

You can replace `data/customer_churn.csv` with your own dataset and update the feature columns in `src/data_loader.py`.

## How to Push to GitHub

```bash
git init
git add .
git commit -m "Add predictive modeling machine learning project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY` with your GitHub account and repository name.
