# Phishing Website Classification (Python)

This academic project focuses on building machine learning models to classify websites as **phishing**, **legitimate**, or **suspicious**, based on a structured dataset of web attributes.

## 📊 Dataset

- Original dataset: [UCI Phishing Websites Dataset](https://archive.ics.uci.edu/ml/datasets/phishing+websites)
- 1353 observations
- 10 attributes such as:
  - `SFH`, `popUpWindow`, `SSLfinal_State`, `Request_URL`, `web_traffic`, `age_of_domain`, `having_IP_Address`, etc.

## 🔧 My Contribution

This was a collaborative university project.  
My responsibilities included:

- Implementation of all machine learning models **except Classification Tree**, which was handled by my teammate
- Developed:
  - **Random Forest**
  - **Gradient Boosting**
  - **Support Vector Machines (SVM)**
  - **Neural Networks**
  - **Ensemble Methods**
- Performed cross-validation, model comparison, and metric tracking
- Wrote the full Jupyter Notebook code for training/testing pipelines (`phishing.ipynb`)

EDA, correlation analysis, PCA, SelectFromModel feature selection, and Classification Tree modeling were conducted by my teammate.


## 🧠 Models & Evaluation

Applied ML models on various versions of the dataset:
- Full feature set
- Feature set after correlation filtering
- PCA-reduced features
- Features selected by `SelectFromModel`

**Evaluation Metric:** Accuracy (via cross-validation)

| Model            | Notes                          |
|------------------|-------------------------------|
| Decision Tree    | With pruning and feature variants |
| SVM              | RBF kernel                    |
| Random Forest    | Multiple depths tested        |
| Neural Networks  | Baseline model                |
| Ensemble         | Combined models for comparison |

## 🛠️ Tools & Libraries

- Python (Jupyter Notebook)
- Scikit-learn
- Pandas
- Matplotlib / Seaborn

## 📁 Files

- `phishing.ipynb` — full notebook with training, evaluation, and results
- `report.pdf` — report of the project

## 📝 Notes

This project was developed jointly with a classmate.  
I was responsible for the implementation, tuning, and evaluation of all machine learning models except the Classification Tree, which was handled by my teammate.
My work included model design (SVM, Random Forest, Gradient Boosting, Neural Networks, Ensemble), cross-validation setup, and comparison of prediction accuracy.

