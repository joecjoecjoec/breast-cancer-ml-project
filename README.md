# 🎀 Breast Cancer ML Detection Project

## 🧪 Problem Description

Breast cancer is one of the most common cancers worldwide. Early detection significantly improves patient survival rates, but diagnosis often requires time-consuming manual evaluation by medical specialists.

This project aims to build a machine learning classifier that automatically predicts whether a breast tumor is:

- benign (non-cancerous)
- malignant (cancerous)

using 30 numerical diagnostic features extracted from digitized breast imaging data.

The goal is to provide a lightweight, reproducible, and portable decision-support tool that:

- assists practitioners or automated systems in screening patients
- speeds up diagnostic workflows
- reduces manual workload
- offers consistent prediction results across environments

The model and prediction pipeline can be executed directly on a local machine or inside a Docker container, ensuring portability and reproducibility.

⸻


## 📂 Project Structure

```text
breast-cancer-ml-project/
|
├── image/                      # EDA visualizations
│   ├── area_mean.png
│   ├── class_distribution.png
│   ├── correlation.png
│   ├── feature_importance.png
│   └── radius_mean.png
|
├── api.py                      # Flask API used for Docker/Render deployment
├── breast-cancer.csv           # dataset
├── train.py                    # training script (produces model_rf.bin)
├── predict.py                  # local prediction script
├── model_rf.bin                # trained Random Forest model
├── Notebook.ipynb              # EDA + model exploration notebook
|
├── requirements.txt            # dependency list
├── Dockerfile                  # container build instructions
├── .gitignore
└── README.md
```
⸻

## 📊 Dataset

The dataset used for this project is the Breast Cancer Diagnostic Dataset, available on Kaggle:

🔗 Source:
https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

This dataset contains diagnostic measurements extracted from digitized images of breast masses, aiming to classify tumors as benign or malignant based on numeric features.

⸻

## 🧬 Dataset Features

Each row describes one breast tumor using 30 numerical features derived from a digitized image of a fine-needle aspirate (FNA) of a breast mass.

Features include:
	•	Radius (mean distance from center to points on perimeter)
	•	Texture (standard deviation of gray-scale values)
	•	Perimeter, area, smoothness, compactness, concavity, symmetry, etc.
	•	All features are numeric and standardized.

Target Variable:
	•	diagnosis
	•	M → malignant
	•	B → benign

⸻

## 📥 How to Download the Dataset

You can download the dataset in two ways:

### 🔹 Option 1: Download directly from Kaggle UI

1. Go to the dataset page:  
   https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset  
2. Click **Download**  
3. Unzip the file  
4. Place `breast-cancer.csv` into your project folder.


### 🔹 Option 2: Download using Kaggle API (Recommended)


If you have Kaggle API installed:

```bash
pip install kaggle
```

Login first (only needed once):

```bash
kaggle datasets download -d yasserh/breast-cancer-dataset
```

Then unzip: 

```bash
unzip breast-cancer-dataset.zip
```


### 📌 Note

For evaluation and reproducibility, the dataset file breast-cancer.csv is already included directly in this repository, so reviewers do not need to download anything manually.

⸻

## 🧩 Dependency & Environment Management

This project uses a dedicated Python virtual environment to ensure full reproducibility and consistent execution across systems.

### 🔹 Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 📥  Install dependencies

```bash
pip install -r requirements.txt
```



## 📊 Exploratory Data Analysis (EDA)

To better understand the dataset and identify patterns relevant to breast cancer diagnosis, several exploratory analyses were performed.

Below is a summary of the key EDA findings.


### 1. Class Distribution

The dataset shows a **moderate imbalance** between the two classes:

- **Benign (B)**: majority class  
- **Malignant (M)**: minority class  

📌 *This matters because imbalance can affect model performance.*

![Class Distribution](image/class_distribution.png)


### 2. Feature Distributions

The 30 diagnostic features were visualized using histograms.

**Key observations:**

- Many features (e.g., radius, texture, area) show clearly different distributions for malignant vs benign tumors.
- Malignant tumors typically have larger radius, perimeter, and area.
- Some features are skewed and may benefit from scaling or normalization.

![Distribution of radius_mean](image/radius_mean.png)
![Distribution of area_mean](image/area_mean.png)

Malignant tumors (1) generally exhibit higher radius_mean and area_mean values.

---

### 3. Correlation Heatmap

A correlation heatmap was generated to identify relationships between features.

![Correlation Analysis](image/correlation.png)

**Important findings:**

- Strong feature clusters exist (e.g., radius, perimeter, area).
- High correlations suggest that tree-based models like Random Forest can use redundancy effectively.

---

### 4. Feature Importance

After training the Random Forest model, feature importance values were extracted.

**Top contributing features typically include:**

- `radius_mean`
- `perimeter_mean`
- `concavity_mean`
- `area_mean`
- `concave_points_mean`

📌 *These features are biologically meaningful and align with medical understanding.*

![Feature Importance](image/feature_importance.png)


## 🤖 Model Training Logic

The training process has two main stages:

1. Model exploration and selection in `Notebook.ipynb`
2. Final training and model export in `train.py`


### 1. Model exploration & selection (`Notebook.ipynb`)

In the notebook, several experiments were performed to understand how different models behave on this dataset.

**Models compared:**

- Logistic Regression (linear baseline)
- Random Forest Classifier (tree-based, non-linear)

**Evaluation on the validation set used:**

- Accuracy
- ROC–AUC
- Feature importance (for tree-based models)

**Result:**

- Random Forest achieved slightly better performance than Logistic Regression on the validation set (higher ROC–AUC and accuracy).
- Tree-based models also handle correlated features better and do not require feature scaling.

➡️ Therefore, Random Forest was selected as the final model family, and its hyperparameters were tuned in the notebook (e.g. `n_estimators`, `max_depth`, `min_samples_leaf`).



### 2. Final training pipeline (`train.py`)

The file `train.py` contains a minimal, reproducible training pipeline.

#### Step 1 – Load the data

```python
df = pd.read_csv("breast-cancer.csv")
```

#### Step 2 – Preprocess the data

	•	Normalize column names:
	•	convert to lowercase
	•	replace spaces and dashes with underscores _
	•	Encode the target:
	•	diagnosis → target
	•	M → 1 (malignant)
	•	B → 0 (benign)
	•	Drop non-informative columns:
	•	id
	•	diagnosis (original target column)

#### Step 3 – Define features and target

```python
X = df.drop(columns=["target"])
y = df["target"]
```

#### Step 4 – Train the final Random Forest model

The final model uses the best hyperparameters found during notebook exploration:

```python
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
model.fit(X, y)
```

#### Step 5 – Save the trained model

The fitted model is serialized to model_rf.bin so it can be reused by other scripts:

```python
with open("model_rf.bin", "wb") as f:
    pickle.dump(model, f)
```

This exported model is later loaded by:
	•	predict.py for local batch predictions
	•	api.py for the Flask API (Docker + cloud deployment)



## 📝 Exporting Notebook to Script

The Jupyter notebook (`Notebook.ipynb`) is used for exploratory data analysis (EDA),
feature inspection, and trying different machine learning models.

To make the training fully reproducible and suitable for automation, the final
training pipeline was exported into a standalone Python script:

**train.py**
- loads the dataset  
- applies the same preprocessing as in the notebook  
- trains the final Random Forest model  
- saves the model into `model_rf.bin`  

This allows the model to be trained consistently across different environments,
including Docker and cloud hosting.


## 🏋️‍♀️ Train the Model

```bash
python3 train.py
```


## 🔮 Run Predictions

```bash
python3 predict.py
```

## 🐳 Containerization (Docker + Flask API)

The model is deployed as a Flask web service running inside a Docker container.

### 🛠️ Build the Container

```bash
docker build -t breast-cancer-api .
```

### 🚀 Run the API Service

```bash
docker run -p 5001:5000 breast-cancer-api
```

If successful, the terminal will show something like:

* Running on http://0.0.0.0:5000

The API will then be available at:

http://localhost:5001/predict

### ⚠️ Important Note (GET vs POST)

The predict endpoint only accepts POST requests.

If you open it in the browser (GET request), you will see:

Method Not Allowed (405)

This is expected and correct.

To get predictions, you must send a POST request with JSON data (see below).

### 📤 Send a POST Request

Example using curl:

```bash
curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{"features": [14.5,20.4,96.7,600.0,0.12,0.09,0.07,0.05,0.18,0.07,
0.35,1.5,2.4,30.0,0.006,0.02,0.03,0.01,0.02,0.004,
16.2,28.0,110.0,800.0,0.14,0.20,0.23,0.11,0.28,0.08]}'
```

Example output:
```json
{"malignant_probability": 0.18}
```    



## ☁️ Cloud Deployment (Render)

The model is successfully deployed on Render using a Docker container.

### ✅ Live API Endpoint

You can access the deployed API here:

👉 https://breast-cancer-ml-project.onrender.com  
👉 https://breast-cancer-ml-project.onrender.com/predict

### ✅ Example cURL Request

```bash
curl -X POST https://breast-cancer-ml-project.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]}'
```

### ✅ Example Response
```bash
{
  "malignant_probability": 0.54
}
```

## ♻️ Reproducibility

This project is fully reproducible:

- All analysis steps in `Notebook.ipynb` were exported into standalone Python scripts (`train.py`, `predict.py`).
- A dedicated virtual environment is used to ensure consistent execution.
- All dependencies are listed in `requirements.txt`.
- Anyone can reproduce the results by:

```bash
python3 train.py
python3 predict.py
```
- The Dockerfile allows consistent deployment across systems.

