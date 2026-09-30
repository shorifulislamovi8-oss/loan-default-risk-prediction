# Loan / Credit Default Risk Prediction

## Project Overview

This project explores how machine learning can help identify customers who may default on credit card payments. It compares Logistic Regression and Random Forest, with model selection and classification thresholds considered alongside the business impact of false positives and false negatives.

## Business Problem

A false negative is a customer who defaults but is not flagged; a false positive is a customer who is flagged but does not default. These outcomes can carry different business consequences. This project demonstrates threshold selection using an assumed cost of 1 unit per false positive and 5 units per false negative. These are demonstration assumptions, not measured costs for a bank.

## Dataset

The project uses the **Default of Credit Card Clients** dataset, containing 30,000 rows. The target is `default`:

| Target value | Meaning | Share |
| --- | --- | ---: |
| 0 | No Default | 77.88% |
| 1 | Default | 22.12% |

The workbook is expected at `data/raw/default of credit card clients.xls`.

## Project Workflow

1. Business Problem
2. Dataset Understanding
3. Data Cleaning
4. EDA
5. Feature Engineering
6. Preprocessing
7. Train / Validation / Test Split
8. Logistic Regression
9. Random Forest
10. ROC-AUC Evaluation
11. Threshold Optimization
12. Business Cost Optimization
13. Final Test Evaluation
14. Real-world Prediction
15. Limitations

## Key EDA Findings

- Lower `LIMIT_BAL` groups had higher observed default rates.
- Repayment history variables `PAY_0` to `PAY_6` were strong signals.
- `PAY_0` had the strongest observed correlation with default: **0.324794**.
- Customers who defaulted generally had lower payment amounts.
- Demographic variables showed weaker relationships than repayment history.

## Feature Engineering

The project creates `PAY_RATIO1` from `PAY_AMT1` and `BILL_AMT1`:

```text
PAY_RATIO1 = PAY_AMT1 / (abs(BILL_AMT1) + 1)
```

The `+ 1` avoids division by zero when the bill amount is zero.

## Models Used

- **Logistic Regression**, with standardized features and balanced class weights.
- **Random Forest**, with balanced class weights.

Both models are evaluated using ROC-AUC. Thresholds are selected separately using validation-set business cost.

## Train / Validation / Test Strategy

The 30,000 rows are divided into stratified subsets:

| Split | Rows |
| --- | ---: |
| Train | 18,000 |
| Validation | 6,000 |
| Test | 6,000 |

The validation set is used for model comparison and threshold optimization. The test set is held out for final evaluation after selecting the Random Forest and its threshold.

## Model Comparison

| Model | Validation ROC-AUC | Best validation threshold | Minimum validation cost |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.7255474901029193 | 0.52 | 3690 units |
| Random Forest | 0.7722719188346658 | 0.24 | 3339 units |

The Random Forest had the higher validation ROC-AUC and lower minimum validation cost, so it was selected.

## Threshold and Business Cost Optimization

For demonstration, the cost calculation is:

```text
Business cost = (False Positives × 1) + (False Negatives × 5)
```

The threshold is the score cutoff used to flag a customer as a predicted default. Lowering it can capture more defaults while also flagging more customers who do not default. The costs and selected thresholds are specific to this dataset and the stated assumptions; they should not be treated as established bank economics.

## Final Test Results

The selected Random Forest was evaluated on the untouched test set at a threshold of **0.24**.

| Metric | Result |
| --- | ---: |
| ROC-AUC | 0.7605471699969247 |
| Accuracy | 0.60 |
| Default precision | 0.33 |
| Default recall | 0.79 |
| Default F1-score | 0.46 |
| Final business cost | 3539 units |

The high default recall comes with many false positives, reflected in the test results below. The demonstration cost weights penalize false negatives more heavily, but do not represent actual operating costs.

## Confusion Matrix Explanation

| Actual / Predicted | No Default (0) | Default (1) |
| --- | ---: | ---: |
| No Default (0) | TN = 2529 | FP = 2144 |
| Default (1) | FN = 279 | TP = 1048 |

- **True Negative (TN):** correctly identified a customer who did not default.
- **False Positive (FP):** flagged a customer who did not default.
- **False Negative (FN):** did not flag a customer who defaulted.
- **True Positive (TP):** correctly flagged a customer who defaulted.

Using the demonstration costs, the final test cost is `2144 × 1 + 279 × 5 = 3539` units.

## Real-world Prediction Example

A sample customer received a model risk score of **0.81** and was flagged as **High Risk** because the score exceeded the selected threshold of **0.24**. The score should not be interpreted as a guaranteed real-world 81% probability: probability calibration was not separately validated.

## Limitations

- The historical dataset may not generalize perfectly to future customers.
- The target is moderately imbalanced: 22.12% of observations are defaults.
- The selected threshold improves recall but creates many false positives.
- The false-positive cost of 1 and false-negative cost of 5 are demonstration assumptions, not real bank costs.
- Model probabilities were not independently calibrated.
- The model should support human and business review rather than automatically approve or reject loans.

## Project Structure

```text
loan_default_project/
├── .gitignore
├── README.md
├── main.py
├── requirements.txt
└── data/
    └── raw/
        └── default of credit card clients.xls
```

## Installation

From the project root, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## How to Run

From the project root, run:

```bash
python main.py
```

The script expects the dataset at `data/raw/default of credit card clients.xls`.

## Technologies Used

- Python
- pandas
- matplotlib
- scikit-learn
- xlrd

## Final Conclusion

This project developed a credit default risk prediction system using Logistic Regression and Random Forest.

A professional Train / Validation / Test workflow was used to avoid selecting the model or decision threshold on the final test set.

Random Forest achieved the stronger validation performance and was selected as the final model. Using the validation set, the business-cost optimized threshold was 0.24.

On the untouched test set, the final model achieved:

- ROC-AUC: 0.7605
- Default Recall: 79%
- Default Precision: 33%
- Default F1-score: 0.46
- True Positives: 1048
- False Negatives: 279
- False Positives: 2144
- True Negatives: 2529
- Final Business Cost: 3539 units

The lower decision threshold helped identify more customers who eventually defaulted, at the cost of producing more false-positive risk flags.

The business-cost values used in this project (FP = 1, FN = 5) are demonstration assumptions rather than real financial costs. In a real lending environment, the threshold should be selected using actual business costs and risk policies.

The model should therefore be used as a risk-support tool for further review rather than as an automatic loan approval or rejection system.
