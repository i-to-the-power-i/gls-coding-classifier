
---

# **GLS Coding Compression Classifier**  
A novel classification algorithm based on **Generalized Lüroth Series (GLS) coding** for data classification.  

## **📌 Overview**  
This repository provides an implementation of the **GLS Coding Compression Classifier**: Check the arxiv report to know more: https://arxiv.org/pdf/2502.12302
 

## **📂 Repository Structure**  

```
📦 GLS_Coding_Classifier
 ├── GLS_Coding_Classifier.py             # Main GLS classification algorithm
 ├── <dataset_name>.txt                   # Dataset files
 ├── <dataset>_hyperparameter_tuning.py   # Hyperparameter tuning scripts
 ├── hyperparameter_tuning_<dataset>.py   # Hyperparameter tuning scripts
 ├── <dataset>_test.py                     # Testing scripts for evaluation
 ├── test_<dataset>.py                     # Testing scripts for evaluation
 ├── README.md                             # This file
 ├── LICENSE                               # Apache-2.0 License
```

## **📊 List of Hyperparameter Tuning and Test Files**  

| Dataset                        | Hyperparameter Tuning File             | Test File         |
|--------------------------------|----------------------------------|------------------|
| **Breast Cancer Wisconsin**    | `BCW_hyperparameter_tuning.py`  | `BCW_test.py`    |
| **Bank Note Authentication**   | `BNA_hyperparameter_tuning.py`  | `BNA_test.py`    |
| **Ionosphere**                 | `ionosphere_hyperparameter_tuning.py`  | `ionosphere_test.py`  |
| **Iris**                       | `iris_hyperparameter_tuning.py`  | `iris_test.py`  |
| **Wine**                       | `hyperparameter_tuning_wine.py`  | `test_wine.py`  |
| **Seeds**                      | `hyperparameter_tuning_seeds.py` | `test_seeds.py`  |
| **Haberman’s Survival**        | `hyperparameter_tuning_haberman.py` | `test_haberman.py` |

## **📌 Installation & Dependencies**  
Ensure you have **Python 3.x** installed along with the required dependencies.

### **Install Required Libraries**  
```bash
pip install numpy pandas scikit-learn
```

## **🚀 Usage**  
### **Step 1: Hyperparameter Tuning**  
Run the hyperparameter tuning script to determine the best threshold for classification.  

Example:  
```bash
python BCW_hyperparameter_tuning.py
```

### **Step 2: Testing the Model**  
Once the optimal threshold is determined, run the test script to evaluate classification performance.  

Example:  
```bash
python BCW_test.py
```

## **📊 Datasets**  
The repository includes several benchmark datasets stored in `.txt` format:  
- **Breast Cancer Wisconsin** (`breast-cancer-wisconsin.txt`)  
- **Bank Note Authentication** (`data_banknote_authentication.txt`)  
- **Ionosphere** (`ionosphere_data.txt`)  
- **Iris** (`iris.txt`)  
- **Wine** (`wine.txt`)  
- **Seeds** (`seeds_dataset.txt`)  
- **Haberman’s Survival** (`haberman.txt`)  



## **📜 License**  
This project is licensed under the **Apache-2.0 License**.  

---
