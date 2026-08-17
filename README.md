# Quantum Kernel-Based Credit Card Fraud Detection

A comparative Quantum Machine Learning (QML) project that investigates the use of quantum kernel methods for credit-card fraud classification.

The project compares a classical RBF Support Vector Machine (SVM) with Quantum Support Vector Classification (QSVC) using different quantum feature maps. It also investigates how model performance and computational cost change as the dataset size increases.

---

## Project Overview

Credit-card fraud detection is a binary classification problem in which transactions are classified as either:

- Legitimate
- Fraudulent

This project explores whether quantum kernel methods can provide useful feature representations for this classification task.

Rather than assuming that the quantum model will outperform the classical model, the project performs a controlled experimental comparison between classical and quantum approaches.

The main research question is:

> **How do quantum feature-map choices and dataset size affect the performance and computational cost of quantum-kernel-based fraud detection compared with a classical RBF-SVM?**

---

## Objectives

The project has the following objectives:

1. Build a classical RBF-SVM baseline.
2. Implement a Quantum Support Vector Classifier (QSVC).
3. Compare different quantum feature maps.
4. Evaluate classification performance using multiple metrics.
5. Investigate the effect of dataset size on model performance.
6. Compare computational runtime between classical and quantum approaches.
7. Generate reproducible tables, graphs, confusion matrices, and evaluation reports.

---

## Methodology

The overall workflow is:

```text
Credit Card Fraud Dataset
            |
            v
    Balanced Sampling
            |
            v
     Train/Test Split
            |
            v
     Standardization
            |
            v
      PCA Reduction
            |
            v
      2-Dimensional Data
            |
      +-----+-----+
      |           |
      v           v
 Classical     Quantum
    SVM          Kernel
      |           |
      |      +----+----+
      |      |    |    |
      |      Z   ZZ  Pauli
      |      |    |    |
      |      +----+----+
      |           |
      +-----+-----+
            |
            v
       Evaluation
            |
            v
 Accuracy / Precision / Recall
 F1 / ROC-AUC / Specificity
 Confusion Matrix / Runtime