import os
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZES = [
    15,
    25,
    50,
    75,
    100
]

RANDOM_STATE = 42

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def prepare_data(
    data,
    samples_per_class
):

    sampled = (
        data.groupby(
            "Class",
            group_keys=False
        )
        .sample(
            n=samples_per_class,
            random_state=RANDOM_STATE
        )
        .reset_index(drop=True)
    )

    X = sampled.drop(
        "Class",
        axis=1
    )

    y = sampled["Class"]

    # Remove time
    X = X.drop(
        "Time",
        axis=1
    )

    # Standardize
    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    # PCA → 2 dimensions
    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE
    )

    X = pca.fit_transform(X)

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate(
    y_test,
    predictions,
    scores
):

    return {

        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "f1_score":
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_test,
                scores
            )
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print(
        "Loading fraud dataset..."
    )

    url = (
        "https://storage.googleapis.com/"
        "download.tensorflow.org/data/creditcard.csv"
    )

    data = pd.read_csv(url)

    all_results = []

    # ========================================================
    # LOOP OVER DATASET SIZES
    # ========================================================

    for samples in SAMPLE_SIZES:

        print("\n" + "=" * 70)

        print(
            f"SAMPLES PER CLASS: {samples}"
        )

        print("=" * 70)

        X_train, X_test, y_train, y_test = (
            prepare_data(
                data,
                samples
            )
        )

        # ====================================================
        # CLASSICAL SVM
        # ====================================================

        classical = SVC(
            kernel="rbf",
            probability=True
        )

        start = time.perf_counter()

        classical.fit(
            X_train,
            y_train
        )

        train_time = (
            time.perf_counter() - start
        )

        start = time.perf_counter()

        classical_predictions = (
            classical.predict(X_test)
        )

        prediction_time = (
            time.perf_counter() - start
        )

        classical_scores = (
            classical.predict_proba(X_test)[:, 1]
        )

        metrics = evaluate(
            y_test,
            classical_predictions,
            classical_scores
        )

        all_results.append({

            "samples_per_class":
                samples,

            "total_samples":
                samples * 2,

            "model":
                "Classical SVM",

            **metrics,

            "training_time_sec":
                train_time,

            "prediction_time_sec":
                prediction_time
        })

        # ====================================================
        # QUANTUM QSVC
        # ====================================================

        feature_map = zz_feature_map(
            feature_dimension=2,
            reps=2,
            entanglement="linear"
        )

        kernel = FidelityQuantumKernel(
            feature_map=feature_map
        )

        quantum_model = QSVC(
            quantum_kernel=kernel
        )

        start = time.perf_counter()

        quantum_model.fit(
            X_train,
            y_train
        )

        quantum_train_time = (
            time.perf_counter() - start
        )

        start = time.perf_counter()

        quantum_predictions = (
            quantum_model.predict(X_test)
        )

        quantum_prediction_time = (
            time.perf_counter() - start
        )

        quantum_scores = (
            quantum_model.decision_function(
                X_test
            )
        )

        metrics = evaluate(
            y_test,
            quantum_predictions,
            quantum_scores
        )

        all_results.append({

            "samples_per_class":
                samples,

            "total_samples":
                samples * 2,

            "model":
                "Quantum QSVC",

            **metrics,

            "training_time_sec":
                quantum_train_time,

            "prediction_time_sec":
                quantum_prediction_time
        })

        print(
            f"Classical accuracy: "
            f"{all_results[-2]['accuracy']:.4f}"
        )

        print(
            f"Quantum accuracy: "
            f"{all_results[-1]['accuracy']:.4f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    results = pd.DataFrame(
        all_results
    )

    path = os.path.join(
        RESULTS_DIR,
        "scaling_experiment.csv"
    )

    results.to_csv(
        path,
        index=False
    )

    print(
        f"\nSaved results to {path}"
    )


if __name__ == "__main__":
    main()