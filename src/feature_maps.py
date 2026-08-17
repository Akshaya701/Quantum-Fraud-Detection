import os
import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

from qiskit.circuit.library import (
    z_feature_map,
    zz_feature_map,
    pauli_feature_map
)

from qiskit_machine_learning.kernels import (
    FidelityQuantumKernel
)

from qiskit_machine_learning.algorithms import (
    QSVC
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
    # Balanced sampling
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

    print(
        f"Selected {SAMPLES_PER_CLASS} samples per class."
    )

    print(
        f"Total samples: {len(data)}"
    )

    # --------------------------------------------------------
    # Separate features and target
    # --------------------------------------------------------

    X = data.drop(
        "Class",
        axis=1
    )

    y = data["Class"]

    # Time is removed because we don't want the
    # transaction timestamp to act as a feature.
    X = X.drop(
        "Time",
        axis=1
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Split BEFORE fitting scaler and PCA.
    #
    # This prevents test-set information from leaking
    # into preprocessing.
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # PCA
    #
    # We need 2 features because our quantum circuit
    # uses 2 qubits.
    # --------------------------------------------------------

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_test_pca = pca.transform(
        X_test_scaled
    )

    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )

    print(
        f"PCA explained variance: "
        f"{explained_variance:.4f}"
    )

    return (
        X_train_pca,
        X_test_pca,
        y_train,
        y_test
    )


# ============================================================
# EVALUATE ONE FEATURE MAP
# ============================================================

def evaluate_feature_map(
    name,
    feature_map,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 65)
    print(f"FEATURE MAP: {name}")
    print("=" * 65)

    # --------------------------------------------------------
    # Quantum kernel
    # --------------------------------------------------------

    kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    # --------------------------------------------------------
    # QSVC
    # --------------------------------------------------------

    model = QSVC(
        quantum_kernel=kernel
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.perf_counter() - start_time
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    start_time = time.perf_counter()

    predictions = model.predict(
        X_test
    )

    prediction_time = (
        time.perf_counter() - start_time
    )

    # --------------------------------------------------------
    # Decision scores
    # Used for ROC-AUC
    # --------------------------------------------------------

    decision_scores = model.decision_function(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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
        decision_scores
    )

    # --------------------------------------------------------
    # Confusion matrix
    #
    # [[TN, FP],
    #  [FN, TP]]
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # Specificity
    #
    # Specificity = TN / (TN + FP)
    # --------------------------------------------------------

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nAccuracy          : {accuracy:.4f}"
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

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    report_path = os.path.join(
        RESULTS_DIR,
        f"{safe_name}_classification_report.txt"
    )

    with open(
        report_path,
        "w"
    ) as file:

        file.write(
            f"Feature Map: {name}\n\n"
        )

        file.write(report)

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"{name} - Confusion Matrix"
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

    # Add values inside cells
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
            f"{safe_name}_confusion_matrix.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Save feature-map circuit
    # --------------------------------------------------------

    try:

        feature_map.draw(
            output="mpl",
            filename=os.path.join(
                RESULTS_DIR,
                f"{safe_name}_circuit.png"
            )
        )

    except Exception as error:

        print(
            f"Could not save circuit image: {error}"
        )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "Feature Map": name,

        "Samples/Class":
            SAMPLES_PER_CLASS,

        "Total Samples":
            SAMPLES_PER_CLASS * 2,

        "Accuracy":
            accuracy,

        "Balanced Accuracy":
            balanced_accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1 Score":
            f1,

        "ROC-AUC":
            roc_auc,

        "Specificity":
            specificity,

        "True Negatives":
            tn,

        "False Positives":
            fp,

        "False Negatives":
            fn,

        "True Positives":
            tp,

        "Training Time (s)":
            training_time,

        "Prediction Time (s)":
            prediction_time
    }


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.perf_counter()

    # --------------------------------------------------------
    # Load and preprocess
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_data()

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    # ========================================================
    # DEFINE FEATURE MAPS
    # ========================================================

    feature_maps = {

        "Z Feature Map":
            z_feature_map(
                feature_dimension=2,
                reps=2
            ),

        "ZZ Feature Map":
            zz_feature_map(
                feature_dimension=2,
                reps=2,
                entanglement="linear"
            ),

        "Pauli Feature Map":
            pauli_feature_map(
                feature_dimension=2,
                reps=2,
                entanglement="linear",
                paulis=[
                    "Z",
                    "ZZ",
                    "X"
                ]
            )
    }

    # ========================================================
    # RUN ALL FEATURE MAPS
    # ========================================================

    results = []

    for name, feature_map in feature_maps.items():

        result = evaluate_feature_map(
            name,
            feature_map,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(
            result
        )

    # ========================================================
    # CREATE RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # PRINT FINAL TABLE
    # ========================================================

    print(
        "\n\n" + "=" * 80
    )

    print(
        "FINAL FEATURE MAP COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE FULL CSV
    # ========================================================

    results_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "feature_map_comparison.csv"
        ),
        index=False
    )

    # ========================================================
    # CLEAN REPORT TABLE
    # ========================================================

    report_table = results_df[
        [
            "Feature Map",
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "Specificity"
        ]
    ].copy()

    # Round only metric columns
    metric_columns = [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Specificity"
    ]

    report_table[
        metric_columns
    ] = report_table[
        metric_columns
    ].round(4)

    # ========================================================
    # SAVE CLEAN CSV
    # ========================================================

    report_table.to_csv(
        os.path.join(
            RESULTS_DIR,
            "feature_map_comparison_clean.csv"
        ),
        index=False
    )

    # ========================================================
    # CREATE TABLE IMAGE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(14, 4)
    )

    ax.axis("off")

    table = ax.table(
        cellText=report_table.values,
        colLabels=report_table.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        9
    )

    table.scale(
        1,
        2
    )

    plt.title(
        "Quantum Feature Map Performance Comparison",
        fontsize=14,
        pad=20
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "feature_map_comparison_table.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # PERFORMANCE GRAPH
    # ========================================================

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    plot_data = (
        report_table[
            ["Feature Map"] + metrics
        ]
        .set_index("Feature Map")
    )

    ax = plot_data.plot(
        kind="bar",
        figsize=(12, 6)
    )

    ax.set_xlabel(
        "Quantum Feature Map"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_title(
        "Performance Comparison of Quantum Feature Maps"
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        title="Metric"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "feature_map_metrics.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # RUNTIME GRAPH
    # ========================================================

    runtime_data = results_df[
        [
            "Feature Map",
            "Training Time (s)",
            "Prediction Time (s)"
        ]
    ].set_index(
        "Feature Map"
    )

    ax = runtime_data.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_xlabel(
        "Quantum Feature Map"
    )

    ax.set_ylabel(
        "Time (seconds)"
    )

    ax.set_title(
        "Runtime Comparison of Quantum Feature Maps"
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        title="Runtime"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "feature_map_runtime.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_time = (
        time.perf_counter() - overall_start
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Total experiment time: "
        f"{total_time:.2f} seconds"
    )

    print(
        "\nSaved files:"
    )

    print(
        "  results/feature_map_comparison.csv"
    )

    print(
        "  results/feature_map_comparison_clean.csv"
    )

    print(
        "  results/feature_map_comparison_table.png"
    )

    print(
        "  results/feature_map_metrics.png"
    )

    print(
        "  results/feature_map_runtime.png"
    )

    print(
        "  results/*_confusion_matrix.png"
    )

    print(
        "  results/*_classification_report.txt"
    )

    print(
        "  results/*_circuit.png"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()