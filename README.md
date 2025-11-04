# Titanic-Survival-Prediction

### 🧾 Project Overview

This project is a **Supervised Machine Learning Classification Algorithms**. The goal is to implement, train, and evaluate these models on a common classification task to determine the best-performing algorithm for the chosen dataset.

### 🔐 Algorithms Implemented

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Naive Bayes (GaussianNB)**
- **Support Vector Machine (SVM)**

---

### 📂 Project Structure & Usage

The core of the project is contained within four Jupyter Notebooks, each focusing on one algorithm:

|          File Name           |                                 Description                                 |
| :--------------------------: | :-------------------------------------------------------------------------: |
|         `KNN.ipynb`          |  Implementation and evaluation of the **K-Nearest Neighbors** classifier.   |
|  `LogicticRegression.ipynb`  |  Implementation and evaluation of the **Logistic Regression** classifier.   |
|     `Naive_Bayes.ipynb`      |  Implementation and evaluation of the **Gaussian Naive Bayes** classifier.  |
| `SupportVectorMachine.ipynb` | Implementation and evaluation of the **Support Vector Machine** classifier. |

### ✅ How to Run

1.  **Clone the repository:**

    ```bash
    git clone [YOUR_REPOSITORY_LINK]
    cd [YOUR_PROJECT_FOLDER]
    ```

2.  Open and execute the cells in the notebooks (`.ipynb` files) to reproduce the analysis.

---

### 📊 Dataset & Preprocessing

- **Dataset Used**: titanic dataset
- **Source**: seaborn

### Key Preprocessing Steps

All models utilize a consistent set of preprocessing steps to ensure a fair comparison:

1.  **Feature Scaling**: **`StandardScaler`** was applied to standardize features, which is crucial for distance-based algorithms like KNN and SVM.
2.  **Class Balancing**: **`RandomOverSampler`** from `imblearn` was used on the training set to address class imbalance and prevent model bias.

---

### 📈 Model Performance Evaluation

This table summarizes the performance metrics of all four models on the **test set**.

|           Model            | Accuracy | Precision (Avg.) | Recall (Avg.) | F1-Score (Avg.) |
| :------------------------: | :------: | :--------------: | :-----------: | :-------------: |
|  **Logistic Regression**   | **1.00** |       1.00       |     1.00      |      1.00       |
|  **K-Nearest Neighbors**   | **1.00** |       1.00       |     1.00      |      1.00       |
|      **Naive Bayes**       | **1.00** |       1.00       |     1.00      |      1.00       |
| **Support Vector Machine** | **1.00** |       1.00       |     1.00      |      1.00       |

---

### 👇 Install required packages

pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
