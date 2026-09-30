"""Loan Default Risk Prediction

Install dependencies:
python -m pip install pandas xlrd matplotlib scikit-learn

Run:
python main.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "raw"
    / "default of credit card clients.xls"
)

FP_COST = 1
FN_COST = 5


# ==========================================
# Business Cost Function
# ==========================================
def calculate_business_cost(
    y_true,
    y_pred,
    fp_cost=FP_COST,
    fn_cost=FN_COST,
):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    cost = (
        fp * fp_cost
        + fn * fn_cost
    )

    return cost, tn, fp, fn, tp


# ==========================================
# Threshold Optimization Function
# ==========================================
def find_best_threshold(
    y_true,
    probabilities,
    fp_cost=FP_COST,
    fn_cost=FN_COST,
):
    best_threshold = None
    best_cost = float("inf")

    for threshold_percent in range(10, 91):
        threshold = threshold_percent / 100
        predictions = (
            probabilities >= threshold
        ).astype(int)

        cost, _, _, _, _ = (
            calculate_business_cost(
                y_true,
                predictions,
                fp_cost,
                fn_cost,
            )
        )

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold, best_cost


def main():

    # ==========================================
    # 1. Load Dataset
    # ==========================================
    df = pd.read_excel(
        DATA_PATH,
        header=1,
        engine="xlrd",
    )

    df = df.rename(
        columns={
            "default payment next month": "default"
        }
    )

    # ==========================================
    # 2. Data Cleaning
    # ==========================================
    df["EDUCATION"] = df["EDUCATION"].replace(
        {
            0: 4,
            5: 4,
            6: 4,
        }
    )

    df["MARRIAGE"] = df["MARRIAGE"].replace(
        {
            0: 3,
        }
    )

    # ==========================================
    # 3. Feature Engineering
    # ==========================================
    df["PAY_RATIO1"] = (
        df["PAY_AMT1"]
        / (df["BILL_AMT1"].abs() + 1)
    )

    # ==========================================
    # 4. Basic Checks
    # ==========================================
    print("\nSEX values:")
    print(
        df["SEX"]
        .value_counts()
        .sort_index()
    )

    print("\nEDUCATION values:")
    print(
        df["EDUCATION"]
        .value_counts()
        .sort_index()
    )

    print("\nMARRIAGE values:")
    print(
        df["MARRIAGE"]
        .value_counts()
        .sort_index()
    )

    print("\nDefault percentage:")
    print(
        df["default"]
        .value_counts(normalize=True)
        * 100
    )

    # ==========================================
    # 5. Target Distribution
    # ==========================================
    df["default"].value_counts().sort_index().plot(
        kind="bar"
    )

    plt.title("Loan Default Distribution")
    plt.xlabel("Default")
    plt.ylabel("Customer Count")

    plt.xticks(
        [0, 1],
        ["No Default", "Default"],
        rotation=0,
    )

    # Uncomment when needed
    # plt.show()

    # ==========================================
    # 6. AGE Analysis
    # ==========================================
    print("\nAverage AGE by default:")
    print(
        df.groupby("default")["AGE"].mean()
    )

    df["AgeGroup"] = pd.cut(
        df["AGE"],
        bins=[
            20,
            30,
            40,
            50,
            60,
            100,
        ],
        labels=[
            "21-30",
            "31-40",
            "41-50",
            "51-60",
            "60+",
        ],
    )

    print("\nDefault rate by Age Group:")
    print(
        df.groupby(
            "AgeGroup",
            observed=False,
        )["default"].mean()
        * 100
    )

    # ==========================================
    # 7. LIMIT_BAL Analysis
    # ==========================================
    print("\nAverage LIMIT_BAL by default:")
    print(
        df.groupby(
            "default"
        )["LIMIT_BAL"].mean()
    )

    df["LimitGroup"] = pd.cut(
        df["LIMIT_BAL"],
        bins=[
            0,
            50000,
            100000,
            200000,
            300000,
            1000000,
        ],
        labels=[
            "0-50k",
            "50k-100k",
            "100k-200k",
            "200k-300k",
            "300k+",
        ],
    )

    print("\nDefault rate by Limit Group:")
    print(
        df.groupby(
            "LimitGroup",
            observed=False,
        )["default"].mean()
        * 100
    )

    # ==========================================
    # 8. Repayment History Analysis
    # ==========================================
    payment_columns = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]

    for col in payment_columns:

        analysis = (
            df.groupby(col)["default"]
            .agg(["count", "mean"])
        )

        analysis["default_rate"] = (
            analysis["mean"] * 100
        )

        print(f"\n{col} analysis:")

        print(
            analysis[
                [
                    "count",
                    "default_rate",
                ]
            ]
        )

    # ==========================================
    # 9. Bill / Payment Analysis
    # ==========================================
    print("\nAverage BILL_AMT1 by default:")
    print(
        df.groupby(
            "default"
        )["BILL_AMT1"].mean()
    )

    print("\nAverage PAY_AMT1 by default:")
    print(
        df.groupby(
            "default"
        )["PAY_AMT1"].mean()
    )

    print("\nMedian PAY_RATIO1 by default:")
    print(
        df.groupby(
            "default"
        )["PAY_RATIO1"].median()
    )

    positive_bill = (
        df[df["BILL_AMT1"] > 0]
        .copy()
    )

    positive_bill["PAY_RATIO1"] = (
        positive_bill["PAY_AMT1"]
        / positive_bill["BILL_AMT1"]
    )

    print(
        "\nMedian PAY_RATIO1 "
        "for positive bills:"
    )

    print(
        positive_bill.groupby(
            "default"
        )["PAY_RATIO1"].median()
    )

    # ==========================================
    # 10. Demographic Analysis
    # ==========================================
    demographic_columns = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ]

    for col in demographic_columns:

        print(
            f"\nDefault rate by {col}:"
        )

        print(
            df.groupby(
                col
            )["default"].mean()
            * 100
        )

    # ==========================================
    # 11. Correlation Analysis
    # ==========================================
    corr_features = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "PAY_AMT1",
        "PAY_RATIO1",
        "default",
    ]

    print("\nCorrelation with default:")

    print(
        df[corr_features]
        .corr(numeric_only=True)["default"]
        .sort_values(
            ascending=False
        )
    )

    # ==========================================
    # 12. Prepare Features and Target
    # ==========================================
    X = df.drop(
        columns=[
            "default",
            "ID",
            "AgeGroup",
            "LimitGroup",
        ]
    )

    y = df["default"]

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    # ==========================================
    # 13. Train / Validation / Test Split
    # ==========================================

    # First:
    # 80% temporary data
    # 20% final test data
    X_temp, X_test, y_temp, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    # Split temporary data:
    # 75% of 80% = 60% train
    # 25% of 80% = 20% validation
    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.25,
            random_state=42,
            stratify=y_temp,
        )
    )

    print("\nTrain shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    # Expected:
    # Train      = 18000
    # Validation = 6000
    # Test       = 6000

    # ==========================================
    # 14. Scaling for Logistic Regression
    # ==========================================
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ==========================================
    # 15. Logistic Regression
    # ==========================================
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    lr_model.fit(
        X_train_scaled,
        y_train,
    )

    lr_val_proba = lr_model.predict_proba(
        X_val_scaled
    )[:, 1]

    lr_val_auc = roc_auc_score(
        y_val,
        lr_val_proba,
    )

    print(
        "\nLogistic Regression "
        "Validation ROC-AUC:"
    )
    print(lr_val_auc)

    # ==========================================
    # 16. Random Forest
    # ==========================================
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    rf_model.fit(
        X_train,
        y_train,
    )

    rf_val_proba = rf_model.predict_proba(
        X_val
    )[:, 1]

    rf_val_auc = roc_auc_score(
        y_val,
        rf_val_proba,
    )

    print(
        "\nRandom Forest "
        "Validation ROC-AUC:"
    )
    print(rf_val_auc)

    # ==========================================
    # 17. Business Cost Assumptions
    # ==========================================
    print("\nBusiness Cost Assumption:")
    print(f"False Positive Cost = {FP_COST}")
    print(f"False Negative Cost = {FN_COST}")

    # ==========================================
    # 18. Logistic Regression Threshold Search
    # ==========================================
    lr_best_threshold, lr_best_cost = (
        find_best_threshold(
            y_val,
            lr_val_proba,
            FP_COST,
            FN_COST,
        )
    )

    print(
        "\nLogistic Regression "
        "Best Validation Threshold:"
    )
    print(lr_best_threshold)

    print(
        "Logistic Regression "
        "Minimum Validation Cost:"
    )
    print(lr_best_cost)

    # ==========================================
    # 19. Random Forest Threshold Search
    # ==========================================
    rf_best_threshold, rf_best_cost = (
        find_best_threshold(
            y_val,
            rf_val_proba,
            FP_COST,
            FN_COST,
        )
    )

    print(
        "\nRandom Forest "
        "Best Validation Threshold:"
    )
    print(rf_best_threshold)

    print(
        "Random Forest "
        "Minimum Validation Cost:"
    )
    print(rf_best_cost)

    # ==========================================
    # 20. Select Model Using Validation Set
    # ==========================================
    if rf_best_cost < lr_best_cost:

        final_model_name = "Random Forest"
        final_threshold = rf_best_threshold

        final_proba = rf_model.predict_proba(
            X_test
        )[:, 1]

    else:

        final_model_name = (
            "Logistic Regression"
        )

        final_threshold = (
            lr_best_threshold
        )

        final_proba = (
            lr_model.predict_proba(
                X_test_scaled
            )[:, 1]
        )

    print("\nSelected Model:")
    print(final_model_name)

    print("Selected Threshold:")
    print(final_threshold)

    # ==========================================
    # 21. FINAL TEST EVALUATION
    # ==========================================
    final_pred = (
        final_proba >= final_threshold
    ).astype(int)

    final_auc = roc_auc_score(
        y_test,
        final_proba,
    )

    final_cost, tn, fp, fn, tp = (
        calculate_business_cost(
            y_test,
            final_pred,
            FP_COST,
            FN_COST,
        )
    )

    print("\n============================")
    print("FINAL TEST RESULTS")
    print("============================")

    print("\nModel:")
    print(final_model_name)

    print("\nThreshold:")
    print(final_threshold)

    print("\nROC-AUC:")
    print(final_auc)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            final_pred,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            final_pred,
        )
    )

    print("\nTrue Negative:", tn)
    print("False Positive:", fp)
    print("False Negative:", fn)
    print("True Positive:", tp)

    print("\nFinal Business Cost:")
    print(final_cost)



    # ==========================================
    # 22. Real-World Prediction
    # ==========================================
    new_customer = pd.DataFrame(
        [
            {
                "LIMIT_BAL": 50000,
                "SEX": 1,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 30,
                "PAY_0": 2,
                "PAY_2": 2,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "BILL_AMT1": 45000,
                "BILL_AMT2": 42000,
                "BILL_AMT3": 40000,
                "BILL_AMT4": 38000,
                "BILL_AMT5": 35000,
                "BILL_AMT6": 33000,
                "PAY_AMT1": 2000,
                "PAY_AMT2": 2000,
                "PAY_AMT3": 2500,
                "PAY_AMT4": 2500,
                "PAY_AMT5": 3000,
                "PAY_AMT6": 3000,
            }
        ]
    )

    new_customer["PAY_RATIO1"] = (
        new_customer["PAY_AMT1"]
        / (new_customer["BILL_AMT1"].abs() + 1)
    )
    new_customer = new_customer[X.columns]

    new_customer_proba = rf_model.predict_proba(
        new_customer
    )[0, 1]
    new_customer_prediction = int(
        new_customer_proba >= rf_best_threshold
    )

    print("\nNew Customer Default Probability:")
    print(new_customer_proba)

    print("\nNew Customer Risk Prediction:")
    if new_customer_prediction == 1:
        print("High Risk / Likely Default")
    else:
        print("Low Risk / Likely No Default")


if __name__ == "__main__":
    main()