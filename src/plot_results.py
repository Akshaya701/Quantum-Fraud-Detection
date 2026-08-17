import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"


def main():

    path = os.path.join(
        RESULTS_DIR,
        "scaling_experiment.csv"
    )

    data = pd.read_csv(path)

    # ========================================================
    # ACCURACY
    # ========================================================

    plt.figure(figsize=(9, 6))

    for model in data["model"].unique():

        subset = data[
            data["model"] == model
        ]

        plt.plot(
            subset["samples_per_class"],
            subset["accuracy"],
            marker="o",
            label=model
        )

    plt.xlabel(
        "Samples per Class"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Accuracy vs Dataset Size"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "accuracy_vs_dataset_size.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # F1 SCORE
    # ========================================================

    plt.figure(figsize=(9, 6))

    for model in data["model"].unique():

        subset = data[
            data["model"] == model
        ]

        plt.plot(
            subset["samples_per_class"],
            subset["f1_score"],
            marker="o",
            label=model
        )

    plt.xlabel(
        "Samples per Class"
    )

    plt.ylabel(
        "F1 Score"
    )

    plt.title(
        "F1 Score vs Dataset Size"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "f1_vs_dataset_size.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # RECALL
    # ========================================================

    plt.figure(figsize=(9, 6))

    for model in data["model"].unique():

        subset = data[
            data["model"] == model
        ]

        plt.plot(
            subset["samples_per_class"],
            subset["recall"],
            marker="o",
            label=model
        )

    plt.xlabel(
        "Samples per Class"
    )

    plt.ylabel(
        "Recall"
    )

    plt.title(
        "Fraud Recall vs Dataset Size"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "recall_vs_dataset_size.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # TRAINING TIME
    # ========================================================

    plt.figure(figsize=(9, 6))

    for model in data["model"].unique():

        subset = data[
            data["model"] == model
        ]

        plt.plot(
            subset["samples_per_class"],
            subset["training_time_sec"],
            marker="o",
            label=model
        )

    plt.xlabel(
        "Samples per Class"
    )

    plt.ylabel(
        "Training Time (seconds)"
    )

    plt.title(
        "Training Time vs Dataset Size"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "runtime_vs_dataset_size.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "\nAll plots saved in results/"
    )


if __name__ == "__main__":
    main()