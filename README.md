Here’s an improved **README.md** that reflects your repository structure while enhancing clarity and usability.  

---

# **GLS Coding Compression Classifier**  
A novel classification algorithm based on **Generalized Lüroth Series (GLS) coding** for efficient and interpretable data classification.  

## **📌 Overview**  
This repository provides an implementation of the **GLS Coding Compression Classifier**, which utilizes GLS coding to transform data into symbolic sequences, followed by a threshold-based classification approach.  

### **Key Features**  
✔️ Symbolic representation of data using GLS coding  
✔️ Hyperparameter tuning for optimal classification performance  
✔️ Threshold-based classification for improved interpretability  
✔️ Supports multiple benchmark datasets  

## **📂 Repository Structure**  

```
📦 GLS_Coding_Classifier
 ├── GLS_Coding_Classifier.py             # Main GLS classification algorithm
 ├── <dataset_name>.txt                   # Dataset files (e.g., breast-cancer-wisconsin.txt)
 ├── <dataset>_hyperparameter_tuning.py   # Hyperparameter tuning scripts
 ├── <dataset>_test.py                     # Testing scripts for evaluation
 ├── README.md                             # This file
 ├── LICENSE                               # Apache-2.0 License
```

## **📌 Installation & Dependencies**  
Ensure you have **Python 3.x** installed along with the required dependencies.

### **Install Required Libraries**  
```bash
pip install numpy pandas scikit-learn
```

## **🚀 Usage**  
### **Step 1: Hyperparameter Tuning**  
Run the hyperparameter tuning script to determine the best threshold for classification.  

```bash
python <dataset>_hyperparameter_tuning.py
```
Example:  
```bash
python iris_hyperparameter_tuning.py
```

### **Step 2: Testing the Model**  
Once the optimal threshold is determined, run the test script to evaluate classification performance.  

```bash
python <dataset>_test.py
```
Example:  
```bash
python iris_test.py
```

## **📊 Datasets**  
The repository includes several benchmark datasets stored in `.txt` format:  
- **Iris** (`iris.txt`)  
- **Breast Cancer Wisconsin** (`breast-cancer-wisconsin.txt`)  
- **Wine** (`wine.txt`)  
- **Bank Note Authentication** (`data_banknote_authentication.txt`)  
- **Ionosphere** (`ionosphere_data.txt`)  
- **Seeds** (`seeds_dataset.txt`)  
- **Haberman’s Survival** (`haberman.txt`)  

## **🔬 Authors**  
- **Harikrishnan NB**  
- **Anuja Vats**  
- **Nithin Nagaraj**  
- **Marius Pedersen**  

## **📜 License**  
This project is licensed under the **Apache-2.0 License**.  

