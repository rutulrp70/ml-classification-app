# ML Classification Models - Assignment 2

## 📋 Problem Statement

This project implements multiple machine learning classification models to predict binary classification outcomes using a public dataset. The objective is to develop, train, and evaluate six different classification algorithms, compare their performance metrics, and deploy an interactive web application using Streamlit for real-time predictions and model evaluation.

---

## 📊 Dataset Description

**Dataset Name:** Breast Cancer Classification Dataset

**Source:** scikit-learn built-in datasets

**Dataset Characteristics:**
- **Total Samples:** 569 instances
- **Total Features:** 30 numerical features
- **Target Variable:** Binary classification (0: Malignant, 1: Benign)
- **Feature Types:** All numerical (computed from digitized image measurements)
- **Class Distribution:**
  - Negative (0 - Malignant): 212 samples
  - Positive (1 - Benign): 357 samples

**Data Split:**
- Training Set: 455 samples (80%)
- Test Set: 114 samples (20%)

**Feature Descriptions:**
The dataset includes computed measurements of cell characteristics such as:
- Radius, texture, perimeter, area
- Smoothness, compactness, concavity
- Concave points, symmetry, fractal dimension
- And corresponding "worst" values for each measurement

**Preprocessing Steps:**
1. Loaded dataset from scikit-learn
2. Separated features (X) and target (y)
3. Split data into training (80%) and test (20%) sets with stratification
4. Applied StandardScaler for feature normalization

---

## 🤖 Models Used & Evaluation Metrics

### Classification Models Implemented:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbor (k-NN) Classifier**
4. **Naive Bayes Classifier (Gaussian)**
5. **Random Forest Classifier (Ensemble)**
6. **XGBoost Classifier (Ensemble)**

### Evaluation Metrics Computed for Each Model:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correct predictions out of total predictions |
| **AUC Score** | Area Under the Receiver Operating Characteristic Curve (0-1) |
| **Precision** | True Positive / (True Positive + False Positive) |
| **Recall** | True Positive / (True Positive + False Negative) |
| **F1 Score** | Harmonic mean of Precision and Recall (2*Precision*Recall)/(Precision+Recall) |
| **MCC (Matthews Correlation Coefficient)** | Correlation coefficient between predicted and actual binary classifications |

---

## 📈 Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| K-Nearest Neighbor | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| XGBoost (Ensemble) | 0.9561 | 0.9901 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

---

## 📝 Model Performance Analysis & Observations

### Overall Performance
All six models demonstrated excellent performance on the breast cancer classification dataset, with accuracy scores ranging from 91.23% to 98.25%. This indicates that the dataset is well-suited for classification tasks and the chosen features are highly predictive.

### Individual Model Observations

#### 1. **Logistic Regression** ⭐ BEST OVERALL
- **Performance:** Highest accuracy (98.25%) and excellent metrics across all evaluation criteria
- **Strengths:** 
  - Excellent overall performance with AUC of 0.9954
  - Best precision and recall balance (both 0.9861)
  - Highest MCC score (0.9623), indicating excellent quality predictions
  - Simple and interpretable model
- **Weaknesses:** None significant
- **Observations:** This linear model performs exceptionally well, suggesting that the features have strong linear separability for this classification problem.

#### 2. **Decision Tree Classifier**
- **Performance:** Lower accuracy (91.23%) compared to other models
- **Strengths:**
  - Easy to interpret and visualize
  - No feature scaling required
  - Handles non-linear relationships
- **Weaknesses:**
  - Lowest AUC score (0.9157) among all models
  - Prone to overfitting
  - Lower recall (0.9028) compared to other models
- **Observations:** The decision tree shows signs of overfitting. Increasing max_depth parameter could improve generalization, but the current performance is still acceptable. The model performs well but is outperformed by other approaches.

#### 3. **K-Nearest Neighbor (k-NN)**
- **Performance:** Strong accuracy (95.61%), second best after Logistic Regression
- **Strengths:**
  - Good accuracy and balanced precision-recall (both 0.9589-0.9722)
  - High AUC score (0.9788)
  - Non-parametric approach works well with this dataset
  - High MCC score (0.9054)
- **Weaknesses:**
  - Computationally expensive for large datasets
  - Sensitive to feature scaling (mitigated by preprocessing)
  - No explicit feature importance
- **Observations:** K-NN with k=5 neighbors provides reliable and balanced predictions. The model's performance suggests that similar instances in the feature space have similar classification outcomes.

#### 4. **Naive Bayes Classifier** 🌟 BEST AUC
- **Performance:** Good accuracy (92.98%) with excellent AUC (0.9868)
- **Strengths:**
  - Highest AUC score (0.9868), indicating excellent ranking ability
  - Fast training and prediction
  - Works well with high-dimensional data
  - Balanced precision and recall (both 0.9444)
- **Weaknesses:**
  - Assumes feature independence (which may not be true in this dataset)
  - Lower accuracy than Logistic Regression and k-NN
  - Lower MCC score (0.8492)
- **Observations:** Despite assuming feature independence, Naive Bayes performs remarkably well with an excellent AUC score. The model excels at ranking predictions but has slightly lower overall accuracy.

#### 5. **Random Forest Classifier** ⭐ BEST ENSEMBLE METHOD
- **Performance:** Excellent accuracy (95.61%), tied with k-NN and XGBoost
- **Strengths:**
  - Extremely high AUC score (0.9939) - best among all models
  - Handles feature interactions well
  - Provides feature importance information
  - Robust to outliers and non-linear relationships
  - High recall (0.9722) minimizes false negatives
- **Weaknesses:**
  - More complex model (harder to interpret)
  - Requires more computational resources
  - Cannot extrapolate beyond training data range
- **Observations:** Random Forest demonstrates the power of ensemble methods. The model achieves excellent performance by combining multiple decision trees. The AUC score of 0.9939 is exceptional and indicates outstanding ranking ability.

#### 6. **XGBoost Classifier**
- **Performance:** Excellent accuracy (95.61%), matching Random Forest and k-NN
- **Strengths:**
  - Strong AUC score (0.9901)
  - Highest recall score (0.9861) - excellent at identifying positive cases
  - Sequential tree building improves performance
  - Handles class imbalance well
  - High MCC score (0.9058)
- **Weaknesses:**
  - More complex model with many hyperparameters
  - Requires careful tuning to avoid overfitting
  - Computationally more intensive than simpler models
- **Observations:** XGBoost achieves competitive performance with excellent recall, making it ideal for applications where missing positive cases (false negatives) is costly. The model's performance validates its position as one of the most effective gradient boosting implementations.

### Key Findings:

1. **Linear vs Non-linear:** Logistic Regression (linear) outperforms more complex models, indicating that the classification boundary is largely linear in the feature space.

2. **Ensemble Methods:** Both Random Forest (AUC: 0.9939) and XGBoost (AUC: 0.9901) provide excellent AUC scores, though they don't surpass the simpler Logistic Regression in overall accuracy.

3. **Recommended Model:** For this specific dataset, **Logistic Regression** is recommended due to:
   - Highest accuracy (98.25%)
   - Highest MCC score (0.9623)
   - Interpretability and simplicity
   - Excellent all-around performance

4. **Alternative Recommendations:**
   - For production with focus on ranking: **Random Forest** (best AUC)
   - For balance of simplicity and performance: **k-NN**
   - For minimizing false negatives: **XGBoost** (highest recall)

---

## 🎯 Project Structure

```
ml-classification-app/
├── app.py                          # Streamlit web application
├── train_models.py                 # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── breast_cancer.csv           # Dataset file
└── models/
    ├── logistic_regression.pkl     # Trained LR model
    ├── decision_tree.pkl           # Trained DT model
    ├── knn.pkl                     # Trained k-NN model
    ├── naive_bayes.pkl             # Trained NB model
    ├── random_forest.pkl           # Trained RF model
    ├── xgboost.pkl                 # Trained XGBoost model
    ├── scaler.pkl                  # Feature scaler
    ├── results.pkl                 # Evaluation results
    └── evaluation_results.csv      # Results as CSV
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd ml-classification-app
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train Models (Optional - Models are Pre-trained)
```bash
python train_models.py
```

### Step 5: Run the Streamlit App
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

---

## 📱 Streamlit App Features

### 1. **Model Comparison Page**
- View all 6 models' evaluation metrics in a comparison table
- Download results as CSV
- Interactive visualizations:
  - Accuracy comparison bar chart
  - AUC score comparison
  - F1 Score comparison
  - MCC comparison

### 2. **Model Prediction Page**
- Upload CSV files with test data
- Select a specific model for predictions
- Display predictions with probability scores
- Download predictions as CSV
- Shows confidence scores for each prediction

### 3. **Dataset Analysis Page**
- View dataset statistics and descriptions
- Target class distribution (pie chart)
- Sample count visualization
- Feature statistics table

### 4. **Interactive Features**
- Responsive design that works on all devices
- Real-time metric calculations
- Confusion matrix visualization
- ROC curve analysis
- Classification reports

---

## 📊 How to Use the Application

### Making Predictions:
1. Navigate to the "Model Prediction" page
2. Click "Browse files" to upload your test data CSV
3. Select a model from the dropdown
4. Click "Make Predictions"
5. View and download the results

### Comparing Models:
1. Go to the "Model Comparison" page
2. View the metrics table showing all models' performance
3. Interact with the charts to explore model performance
4. Download the comparison as CSV

### Analyzing the Dataset:
1. Visit the "Dataset Analysis" page
2. View dataset statistics
3. Analyze target distribution and feature statistics

---

## 🌐 Deployment on Streamlit Community Cloud

### Prerequisites:
- GitHub account with repository containing the code
- Streamlit Community Cloud account (free)

### Deployment Steps:

1. **Push code to GitHub:**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Visit Streamlit Cloud:**
   - Go to https://streamlit.io/cloud
   - Sign in with your GitHub account

3. **Create New App:**
   - Click "New app"
   - Select your GitHub repository
   - Choose the branch (usually `main`)
   - Select `app.py` as the entry point

4. **Deploy:**
   - Click "Deploy"
   - Wait for the deployment to complete (usually 2-3 minutes)
   - Share the generated URL with others

---

## 🔬 Technical Details

### Libraries Used:
- **pandas & numpy:** Data manipulation and numerical operations
- **scikit-learn:** Machine learning models and metrics
- **xgboost:** Gradient boosting implementation
- **streamlit:** Web application framework
- **plotly:** Interactive visualizations
- **pickle:** Model serialization

### Model Training Configuration:
- **Train-Test Split:** 80-20 with stratification
- **Feature Scaling:** StandardScaler normalization
- **Random State:** 42 (for reproducibility)

### Key Hyperparameters:
- Logistic Regression: max_iter=1000
- Decision Tree: max_depth=15
- k-NN: n_neighbors=5
- Random Forest: n_estimators=100
- XGBoost: n_estimators=100

---

## 📈 Performance Metrics Explained

- **Accuracy:** Overall correctness of the model (TP+TN)/(Total)
- **AUC:** Probability that model ranks random positive higher than negative
- **Precision:** Of predicted positives, how many were actually positive
- **Recall:** Of actual positives, how many did model find
- **F1 Score:** Harmonic mean balancing Precision and Recall
- **MCC:** Correlation between predicted and actual labels (-1 to 1)

---

## 📝 Notes & Observations

1. The Breast Cancer dataset exhibits good class balance, making it suitable for standard classification approaches
2. All models achieved >91% accuracy, indicating strong predictive power
3. Logistic Regression's superior performance suggests largely linear separation of classes
4. Ensemble methods provide alternative solutions with excellent AUC scores
5. Feature scaling was crucial for distance-based and linear models

---

## 🙋 Support

For issues or questions:
1. Check the Streamlit app for data requirements
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Verify CSV format matches the training data features
4. Refer to the "Dataset Analysis" page for feature information

---

**Last Updated:** February 2026  
**Status:** ✅ Complete & Ready for Deployment