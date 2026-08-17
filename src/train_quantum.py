import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLES_PER_CLASS = 50
RANDOM_STATE = 42
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    url = (
        "https://storage.googleapis.com/"
        "download.tensorflow.org/data/creditcard.csv"
    )

    data = pd.read_csv(url)

    # --------------------------------------------------------
    # Balanced sampling
    # --------------------------------------------------------

    data = (
        data.groupby("Class", group_keys=False)
        .sample(
            n=SAMPLES_PER_CLASS,
            random_state=RANDOM_STATE
        )
        .reset_index(drop=True)
    )

    X = data.drop("Class", axis=1)
    y = data["Class"]

    # --------------------------------------------------------
    # Remove Time
    # --------------------------------------------------------

    X = X.drop("Time", axis=1)

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------------
    # PCA
    # Reduce original features to 2 dimensions
    # --------------------------------------------------------

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE
    )

    X_reduced = pca.fit_transform(X_scaled)

    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )

    print(
        f"PCA explained variance: "
        f"{explained_variance:.4f}"
    )

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    return train_test_split(
        X_reduced,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.perf_counter()

    X_train, X_test, y_train, y_test = load_data()

    print("\n" + "=" * 60)
    print("QUANTUM KERNEL FRAUD DETECTION")
    print("=" * 60)

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    # ========================================================
    # QUANTUM FEATURE MAP
    # ========================================================

    feature_map = zz_feature_map(
        feature_dimension=2,
        reps=2,
        entanglement="linear"
    )

    print("\nQuantum Feature Map:")
    print(feature_map)

    # Save circuit diagram
    circuit_path = os.path.join(
        RESULTS_DIR,
        "quantum_feature_map.png"
    )

    try:
        feature_map.draw(
            output="mpl",
            filename=circuit_path
        )
    except Exception:
        pass

    # ========================================================
    # QUANTUM KERNEL
    # ========================================================

    kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    # ========================================================
    # QSVC
    # ========================================================

    model = QSVC(
        quantum_kernel=kernel
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print("\nTraining QSVC...")

    train_start = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    train_time = (
        time.perf_counter() - train_start
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction_start = time.perf_counter()

    predictions = model.predict(
        X_test
    )

    prediction_time = (
        time.perf_counter() - prediction_start
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
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

    # QSVC provides decision_function
    decision_scores = model.decision_function(
        X_test
    )

    roc_auc = roc_auc_score(
        y_test,
        decision_scores
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    total_time = (
        time.perf_counter() - total_start
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1 Score        : {f1:.4f}")
    print(f"ROC-AUC         : {roc_auc:.4f}")

    print(
        f"\nTraining time   : "
        f"{train_time:.4f} sec"
    )

    print(
        f"Prediction time : "
        f"{prediction_time:.4f} sec"
    )

    print(
        f"Total time      : "
        f"{total_time:.4f} sec"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Legitimate",
                "Fraud"
            ],
            zero_division=0
        )
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    results = pd.DataFrame([{

        "model": "Quantum QSVC",

        "samples_per_class":
            SAMPLES_PER_CLASS,

        "total_samples":
            SAMPLES_PER_CLASS * 2,

        "training_samples":
            len(X_train),

        "testing_samples":
            len(X_test),

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1_score":
            f1,

        "roc_auc":
            roc_auc,

        "training_time_sec":
            train_time,

        "prediction_time_sec":
            prediction_time,

        "total_time_sec":
            total_time
    }])

    results.to_csv(
        os.path.join(
            RESULTS_DIR,
            "quantum_results.csv"
        ),
        index=False
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Legitimate",
            "Fraud"
        ]
    )

    disp.plot()

    plt.title(
        "Quantum QSVC Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "quantum_confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # QUANTUM KERNEL MATRIX
    # ========================================================

    print(
        "\nCalculating quantum kernel matrix..."
    )

    kernel_matrix = kernel.evaluate(
        X_train
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            "quantum_kernel_matrix.npy"
        ),
        kernel_matrix
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        kernel_matrix,
        cmap="viridis",
        aspect="auto"
    )

    plt.colorbar(
        label="Kernel Similarity"
    )

    plt.title(
        "Quantum Kernel Matrix"
    )

    plt.xlabel(
        "Training Sample"
    )

    plt.ylabel(
        "Training Sample"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "quantum_kernel_matrix.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "\nResults saved in:"
        f" {RESULTS_DIR}/"
    )


if __name__ == "__main__":
    main()