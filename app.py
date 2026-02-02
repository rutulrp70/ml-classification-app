"""
Streamlit Web Application for ML Classification Models
Interactive interface for training and evaluating multiple classification models
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
import warnings
warnings.filterwarnings('ignore')


# Page Configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_models():
    """Load pre-trained models from disk"""
    models = {}
    model_dir = 'models'
    
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbor': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for model_name, file_name in model_files.items():
        model_path = os.path.join(model_dir, file_name)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
    
    return models


def load_scaler():
    """Load the fitted scaler"""
    scaler_path = 'models/scaler.pkl'
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    return None


def load_evaluation_results():
    """Load pre-computed evaluation results"""
    results_path = 'models/results.pkl'
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            return pickle.load(f)
    return None


def display_metrics(y_true, y_pred, y_pred_proba):
    """Display evaluation metrics in columns"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        accuracy = accuracy_score(y_true, y_pred)
        st.metric("Accuracy", f"{accuracy:.4f}")
    
    with col2:
        auc_score = roc_auc_score(y_true, y_pred_proba)
        st.metric("AUC Score", f"{auc_score:.4f}")
    
    with col3:
        precision = precision_score(y_true, y_pred)
        st.metric("Precision", f"{precision:.4f}")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        recall = recall_score(y_true, y_pred)
        st.metric("Recall", f"{recall:.4f}")
    
    with col5:
        f1 = f1_score(y_true, y_pred)
        st.metric("F1 Score", f"{f1:.4f}")
    
    with col6:
        mcc = matthews_corrcoef(y_true, y_pred)
        st.metric("MCC", f"{mcc:.4f}")


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix as heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Negative', 'Positive'],
        y=['Negative', 'Positive'],
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400
    )
    
    return fig


def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='#0066cc', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#ff0000', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("# 🤖 ML Classification Models Demo")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.markdown("## 📋 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Model Comparison", "Model Prediction", "Dataset Analysis"]
    )
    
    if page == "Model Comparison":
        st.header("Model Comparison & Evaluation")
        
        # Load evaluation results
        results = load_evaluation_results()
        
        if results:
            # Create comparison dataframe
            comparison_data = []
            for model_name, metrics in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics['Accuracy'],
                    'AUC': metrics['AUC'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1 Score': metrics['F1 Score'],
                    'MCC': metrics['MCC']
                })
            
            df_results = pd.DataFrame(comparison_data)
            
            st.subheader("📊 Evaluation Metrics Comparison")
            st.dataframe(df_results, width='stretch')
            
            # Download results
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="model_comparison.csv",
                mime="text/csv"
            )
            
            # Visualize metrics
            st.subheader("📈 Metrics Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                fig_accuracy = px.bar(df_results, x='Model', y='Accuracy', 
                                     title='Accuracy Comparison',
                                     color='Accuracy',
                                     color_continuous_scale='Viridis')
                st.plotly_chart(fig_accuracy, width='stretch')
            
            with col2:
                # AUC comparison
                fig_auc = px.bar(df_results, x='Model', y='AUC',
                                title='AUC Score Comparison',
                                color='AUC',
                                color_continuous_scale='Viridis')
                st.plotly_chart(fig_auc, width='stretch')
            
            col3, col4 = st.columns(2)
            
            with col3:
                # F1 Score comparison
                fig_f1 = px.bar(df_results, x='Model', y='F1 Score',
                               title='F1 Score Comparison',
                               color='F1 Score',
                               color_continuous_scale='Viridis')
                st.plotly_chart(fig_f1, width='stretch')
            
            with col4:
                # MCC comparison
                fig_mcc = px.bar(df_results, x='Model', y='MCC',
                                title='Matthews Correlation Coefficient Comparison',
                                color='MCC',
                                color_continuous_scale='Viridis')
                st.plotly_chart(fig_mcc, width='stretch')
        else:
            st.warning("⚠️ No evaluation results found. Please train the models first.")
    
    elif page == "Model Prediction":
        st.header("Make Predictions with Trained Models")
        
        # Load models and scaler
        models = load_models()
        scaler = load_scaler()
        
        if models and scaler:
            st.subheader("📤 Upload Test Data")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a CSV file with features",
                type="csv",
                help="CSV file should have the same features as the training data"
            )
            
            if uploaded_file is not None:
                try:
                    df_test = pd.read_csv(uploaded_file)
                    
                    st.success(f"✅ File uploaded successfully! Shape: {df_test.shape}")
                    
                    # Display first few rows
                    st.subheader("Data Preview")
                    st.dataframe(df_test.head(), width='stretch')
                    
                    # Model selection
                    selected_model = st.selectbox(
                        "Select a model for prediction:",
                        list(models.keys())
                    )
                    
                    # Make predictions
                    if st.button("🚀 Make Predictions", key="predict_button"):
                        # Scale features
                        df_scaled = scaler.transform(df_test)
                        
                        # Get model
                        model = models[selected_model]
                        
                        # Make predictions
                        predictions = model.predict(df_scaled)
                        probabilities = model.predict_proba(df_scaled)
                        
                        # Display results
                        st.subheader(f"Predictions using {selected_model}")
                        
                        results_df = pd.DataFrame({
                            'Prediction': predictions,
                            'Probability (Class 0)': probabilities[:, 0],
                            'Probability (Class 1)': probabilities[:, 1],
                            'Confidence': np.max(probabilities, axis=1)
                        })
                        
                        st.dataframe(results_df, width='stretch')
                        
                        # Download predictions
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
            else:
                st.info("ℹ️ Please upload a CSV file to make predictions")
        else:
            st.warning("⚠️ Models not found. Please ensure models are trained first.")
    
    elif page == "Dataset Analysis":
        st.header("Dataset Information & Analysis")
        
        # Load dataset
        data_path = 'data/breast_cancer.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", df.shape[0])
            
            with col2:
                st.metric("Total Features", df.shape[1] - 1)
            
            with col3:
                st.metric("Positive Cases", (df['target'] == 1).sum())
            
            with col4:
                st.metric("Negative Cases", (df['target'] == 0).sum())
            
            st.subheader("📊 Dataset Overview")
            st.dataframe(df.describe(), width='stretch')
            
            # Target distribution
            st.subheader("🎯 Target Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                target_counts = df['target'].value_counts()
                fig_dist = px.pie(
                    values=target_counts.values,
                    names=['Negative (0)', 'Positive (1)'],
                    title='Target Distribution',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                )
                st.plotly_chart(fig_dist, width='stretch')
            
            with col2:
                fig_bar = px.bar(
                    x=['Negative (0)', 'Positive (1)'],
                    y=target_counts.values,
                    title='Sample Count by Class',
                    labels={'x': 'Class', 'y': 'Count'},
                    color=target_counts.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_bar, width='stretch')
            
            # Feature statistics
            st.subheader("📈 Feature Statistics")
            st.dataframe(df.describe().T, width='stretch')
            
        else:
            st.warning("⚠️ Dataset file not found.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>ML Classification Models | Built with Streamlit 🎈</p>
        <p>Assignment 2 - Machine Learning Course</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
