"""
Plant Disease Classification Dashboard
Corn Leaf Disease Detection using Deep Learning
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Page config
st.set_page_config(
    page_title="Corn Disease Classifier",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Class names
CLASS_NAMES = ['Cercospora_leaf_spot', 'Common_rust', 'Northern_Leaf_Blight', 'healthy']
CLASS_DISPLAY_NAMES = {
    'Cercospora_leaf_spot': '🌿 Cercospora Leaf Spot',
    'Common_rust': '🟤 Common Rust',
    'Northern_Leaf_Blight': '🍂 Northern Leaf Blight',
    'healthy': '✅ Healthy Leaf'
}

# Load model
@st.cache_resource
def load_trained_model():
    model_path = "models/best_model.keras"
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

# Load evaluation results
@st.cache_data
def load_evaluation_data():
    try:
        metrics_df = pd.read_csv("models/test_metrics.csv")
        predictions_df = pd.read_csv("models/predictions.csv")
        return metrics_df, predictions_df
    except:
        return None, None

def preprocess_image(image):
    """Preprocess uploaded image for prediction"""
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    # Header
    st.markdown('<h1 class="main-header">🌽 Corn Leaf Disease Classification Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1995/1995572.png", width=100)
        st.markdown("## 📊 Navigation")
        page = st.radio("Select Page", ["🏠 Home", "📈 Model Performance", "🔬 Predict Disease", "📚 About"])
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.info("""
        - **Model:** MobileNetV2 (Transfer Learning)
        - **Classes:** 4 Corn Diseases + Healthy
        - **Test Accuracy:** 90.48%
        """)
    
    # Page 1: Home
    if page == "🏠 Home":
        st.markdown("## Welcome to Corn Disease Detection System!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Classes", "4")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Test Accuracy", "90.48%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Training Images", "47,730")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Test Images", "14,524")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🌟 Key Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **✅ Disease Detection**
            - Common Rust
            - Northern Leaf Blight
            - Cercospora Leaf Spot
            - Healthy Leaf Detection
            """)
        
        with col2:
            st.markdown("""
            **✅ Features**
            - Fast & Accurate
            - Easy to Use
            - Real-time Predictions
            - Visual Analytics
            """)
    
    # Page 2: Model Performance
    elif page == "📈 Model Performance":
        st.markdown("## Model Performance Analytics")
        
        # Load confusion matrix
        if os.path.exists("models/confusion_matrix.png"):
            st.image("models/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
        
        # Classification Report
        if os.path.exists("models/classification_report.txt"):
            with open("models/classification_report.txt", "r") as f:
                report = f.read()
            st.text(report)
        
        # Metrics visualization
        st.markdown("### 📊 Class-wise Performance")
        
        # Sample data from classification report
        class_metrics = {
            'Class': ['Cercospora\nLeaf Spot', 'Common Rust', 'Northern\nLeaf Blight', 'Healthy'],
            'Precision': [0.79, 0.95, 0.82, 0.97],
            'Recall': [0.64, 0.99, 0.85, 0.98],
            'F1-Score': [0.71, 0.97, 0.84, 0.97]
        }
        
        metrics_df = pd.DataFrame(class_metrics)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Precision', x=metrics_df['Class'], y=metrics_df['Precision'], marker_color='#4CAF50'))
        fig.add_trace(go.Bar(name='Recall', x=metrics_df['Class'], y=metrics_df['Recall'], marker_color='#2196F3'))
        fig.add_trace(go.Bar(name='F1-Score', x=metrics_df['Class'], y=metrics_df['F1-Score'], marker_color='#FF9800'))
        
        fig.update_layout(title='Class-wise Performance Metrics', barmode='group', xaxis_title='Disease Class', yaxis_title='Score')
        st.plotly_chart(fig, use_container_width=True)
    
    # Page 3: Predict Disease
    elif page == "🔬 Predict Disease":
        st.markdown("## Upload Corn Leaf Image for Disease Detection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=300)
        
        with col2:
            if uploaded_file is not None:
                model = load_trained_model()
                
                if model:
                    with st.spinner("Analyzing image..."):
                        processed_img = preprocess_image(image)
                        predictions = model.predict(processed_img)[0]
                        predicted_class = CLASS_NAMES[np.argmax(predictions)]
                        confidence = np.max(predictions) * 100
                    
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    # Display result with color coding
                    if predicted_class == 'healthy':
                        st.success(f"### 🌟 Prediction: {CLASS_DISPLAY_NAMES[predicted_class]}")
                    else:
                        st.warning(f"### ⚠️ Prediction: {CLASS_DISPLAY_NAMES[predicted_class]}")
                    
                    st.metric("Confidence", f"{confidence:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show all class probabilities
                    st.markdown("### 📊 All Class Probabilities")
                    
                    prob_df = pd.DataFrame({
                        'Disease': [CLASS_DISPLAY_NAMES[c] for c in CLASS_NAMES],
                        'Probability (%)': predictions * 100
                    }).sort_values('Probability (%)', ascending=False)
                    
                    fig = px.bar(prob_df, x='Disease', y='Probability (%)', color='Disease',
                                 title='Prediction Probabilities', color_discrete_sequence=['#4CAF50', '#FF9800', '#F44336', '#2196F3'])
                    fig.update_layout(xaxis_title='Disease Class', yaxis_title='Probability (%)')
                    st.plotly_chart(fig, use_container_width=True)
    
    # Page 4: About
    elif page == "📚 About":
        st.markdown("## About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        This project uses **Deep Learning** to classify Corn Leaf Diseases from images.
        
        ### 🤖 Model Architecture
        - **Base Model:** MobileNetV2 (pre-trained on ImageNet)
        - **Transfer Learning** with Fine-tuning
        - **Input Size:** 224 x 224 pixels
        - **Output:** 4 classes (3 diseases + healthy)
        
        ### 📊 Dataset
        - **Source:** PlantVillage Augmented Dataset (Corn)
        - **Total Images:** 1,04,720 images
        - **Training:** 47,730 images
        - **Validation:** 14,544 images
        - **Test:** 14,524 images
        
        ### 📈 Results
        - **Test Accuracy:** 90.48%
        - **Precision:** 92.04%
        - **Recall:** 89.13%
        
        ### 🛠️ Technologies Used
        - TensorFlow / Keras
        - Python
        - Streamlit (Dashboard)
        - Plotly (Visualizations)
        - scikit-learn (Metrics)
        
        ### 👨‍💻 Developer
        Built as part of MLOps Project for Plant Disease Classification.
        """)

if __name__ == "__main__":
    main()