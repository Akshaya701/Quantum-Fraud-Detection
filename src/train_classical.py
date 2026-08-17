import os
import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLES_PER_CLASS = 50
TEST_SIZE = 0.20
RANDOM_STATE = 42

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    url = (
        "https://storage.googleapis.com/"
        "download.tensorflow.org/data/creditcard.csv"
    )

    print("\nLoading credit-card fraud dataset...")

    data = pd.read_csv(url)

    # --------------------------------------------------------
    # Balanced random sample
    # --------------------------------------------------------

    data = (
        data.groupby(
            "Class",
            group_keys=False
        )
        .sample(
            n=SAMPLES_PER_CLASS,
            random_state=RANDOM_STATE
        )
        .reset_index(drop=True)
    )

    X = data.drop(
        "Class",
        axis=1
    )

    y = data["Class"]

    # Remove Time
    X = X.drop(
        "Time",
        axis=1
    )

    # --------------------------------------------------------
    # Split BEFORE fitting preprocessing
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.perf_counter()

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_data()

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    # ========================================================
    # PREPROCESSING + CLASSIFIER
    # ========================================================

    # StandardScaler and PCA are fitted ONLY on training data.
    #
    # PCA reduces the original feature space to 2 dimensions,
    # matching the two-qubit quantum experiments.

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "pca",
                PCA(
                    n_components=2
                )
            ),

            (
                "classifier",
                SVC(
                    kernel="rbf",
                    probability=True
                )
            )
        ]
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print("\nTraining Classical SVM...")

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.perf_counter() - start
    )

    # ========================================================
    # PREDICT
    # ========================================================

    start = time.perf_counter()

    predictions = model.predict(
        X_test
    )

    prediction_time = (
        time.perf_counter() - start
    )

    # Probability scores for ROC-AUC
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    total_time = (
        time.perf_counter() - total_start
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("CLASSICAL SVM RESULTS")
    print("=" * 60)

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced_accuracy:.4f}"
    )

    print(
        f"Precision         : {precision:.4f}"
    )

    print(
        f"Recall            : {recall:.4f}"
    )

    print(
        f"F1 Score          : {f1:.4f}"
    )

    print(
        f"ROC-AUC           : {roc_auc:.4f}"
    )

    print(
        f"Specificity       : {specificity:.4f}"
    )

    print(
        f"Training Time     : {training_time:.4f} sec"
    )

    print(
        f"Prediction Time   : {prediction_time:.4f} sec"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )

    print(report)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = pd.DataFrame(
        [
            {
                "Model": "Classical SVM",
                "Samples/Class": SAMPLES_PER_CLASS,
                "Total Samples": SAMPLES_PER_CLASS * 2,
                "Accuracy": accuracy,
                "Balanced Accuracy": balanced_accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC-AUC": roc_auc,
                "Specificity": specificity,
                "True Negatives": tn,
                "False Positives": fp,
                "False Negatives": fn,
                "True Positives": tp,
                "Training Time (s)": training_time,
                "Prediction Time (s)": prediction_time,
                "Total Time (s)": total_time
            }
        ]
    )

    results.to_csv(
        os.path.join(
            RESULTS_DIR,
            "classical_results.csv"
        ),
        index=False
    )

    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "Classical SVM Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Legitimate", "Fraud"]
    )

    plt.yticks(
        [0, 1],
        ["Legitimate", "Fraud"]
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "classical_confusion_matrix.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nClassical results saved to results/"
    )


if __name__ == "__main__":
    main()