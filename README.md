# Network Anomaly Detection Using AI and ML

A comprehensive project focusing on detecting anomalies in network traffic to enhance cybersecurity using AI and ML techniques. This project uses the Random Forest algorithm for classification, employing SMOTE for class balancing and hyperparameter tuning for optimization.

---

## 🚀 **Project Overview**
- **Objective:** Detect network anomalies using machine learning.
- **Dataset:** Improved CICIDS2017 and CSECICIDS2018.
- **Model:** Random Forest Classifier.
- **Preprocessing:** Data cleaning, normalization, and encoding using pandas and scikit-learn.
- **Class Balancing:** SMOTE for oversampling minority classes.
- **Evaluation:** Classification report, confusion matrix, precision, recall, and F1-score.
- **Output:** Predicted labels for network traffic (benign or malicious).

---

## 🛠 **Tools and Technologies Used**
- **Programming Language:** Python 3.12
- **Libraries:** Pandas, Scikit-learn, Imbalanced-learn, Joblib
- **Development Environment:** Visual Studio Code
- **Virtual Environment:** Kali Linux and Host Machine (16GB RAM, 8 cores)

---

## ⚙️ **Architecture**

1. **Input Layer:**
    - CSV File (combined_dataset.csv) containing labeled network traffic data.

2. **Processing Layer:**
    - **Preprocessing:** Data Cleaning, Normalization, and Encoding.
    - **Batch Processing:** Data divided into manageable chunks.
    - **Class Balancing:** SMOTE for minority class oversampling.

3. **Model Layer:**
    - Random Forest Classifier with tuned hyperparameters.

4. **Output Layer:**
    - Predicted Labels (Benign or Malicious).
    - Performance Metrics (Precision, Recall, F1-score).

---

## 🔎 **Steps to Execute**

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-repo/network-anomaly-detection.git
    cd network-anomaly-detection
    ```

2. **Create a Virtual Environment and Install Dependencies:**
    ```bash
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```

3. **Run the Project:**
    ```bash
    python main.py
    ```

---

## 📁 **Project Structure**
```bash
.
├── data
│   ├── combined_dataset.csv
├── src
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
├── results
│   ├── classification_report.txt
│   ├── confusion_matrix.png
├── main.py
├── requirements.txt
└── README.md
```

---

## 📊 **Evaluation Metrics**
- **Precision:** 99.95%
- **Recall:** 99.91%
- **F1-Score:** 99.93%

---

## 📧 **Contact**
For any issues or further clarifications, feel free to open an issue on the repository.

---

**Contributors:**
- Moonchild03
- https://github.com/Moonchild03

---

### ⭐️ If you found this project helpful, consider giving it a star!

