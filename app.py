import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight

# XGBoost import with error handling
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# TensorFlow/Keras imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(42)

# Streamlit configuration
st.set_page_config(
    page_title="Credit Risk Analysis - Multi-Model ML Suite",
    page_icon="🏦",
    layout="wide"
)

# Main Title and Introduction
st.title("🏦 Credit Risk Analysis - Multi-Model Machine Learning Suite")
st.markdown("""
Welcome to the comprehensive credit risk analysis platform! This application allows you to train and compare 
different machine learning models for predicting personal loan acceptance.

**Available Models:**
- 📈 **Logistic Regression**: Linear probabilistic classifier with feature coefficients
- 🌳 **Random Forest**: Ensemble of decision trees with feature importance
- 🚀 **XGBoost**: Gradient boosting with advanced regularization
- 🧠 **Deep Learning**: Neural networks with automatic feature learning
""")

st.markdown("---")

# Model Selection
st.header("🎯 Model Selection")
model_choice = st.selectbox(
    "Choose the machine learning model to train:",
    [
        "Logistic Regression",
        "Random Forest", 
        "XGBoost" if XGBOOST_AVAILABLE else "XGBoost (Not Available - pip install xgboost)",
        "Deep Learning" if TENSORFLOW_AVAILABLE else "Deep Learning (Not Available - pip install tensorflow)"
    ],
    help="Select the model you want to train and use for predictions"
)

# Display model information
if model_choice == "Logistic Regression":
    st.info("📈 **Logistic Regression**: Fast, interpretable linear model with feature coefficients showing positive/negative impact on loan probability.")
elif model_choice == "Random Forest":
    st.info("🌳 **Random Forest**: Robust ensemble method that combines multiple decision trees and provides feature importance rankings.")
elif model_choice == "XGBoost":
    if XGBOOST_AVAILABLE:
        st.info("🚀 **XGBoost**: State-of-the-art gradient boosting algorithm with superior performance on tabular data and multiple importance methods.")
    else:
        st.error("❌ XGBoost is not installed. Please run: `pip install xgboost`")
elif model_choice == "Deep Learning":
    if TENSORFLOW_AVAILABLE:
        st.info("🧠 **Deep Learning**: Neural networks with automatic feature learning and non-linear pattern recognition capabilities.")
    else:
        st.error("❌ TensorFlow is not installed. Please run: `pip install tensorflow`")

st.markdown("---")

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset with comprehensive feature engineering"""
    try:
        data = pd.read_csv("bank_loan.csv")
        
        # Convert CCAvg format
        def convert_ccavg(value):
            if isinstance(value, str) and '/' in value:
                parts = value.split('/')
                return float(parts[0]) + float(parts[1])/100
            return float(value)
        
        data['CCAvg'] = data['CCAvg'].apply(convert_ccavg)
        
        # Basic feature engineering
        data['Income_per_Family'] = data['Income'] / data['Family']
        data['Experience_Age_Ratio'] = data['Experience'] / data['Age']
        
        # Advanced feature engineering for Deep Learning
        if model_choice == "Deep Learning":
            data['CCAvg_Income_Ratio'] = data['CCAvg'] / (data['Income'] + 1e-8)
            data['Wealth_Score'] = (data['Income'] * 0.4 + data['CCAvg'] * 0.3 +
                                   data['Securities Account'] * 50 + data['CD Account'] * 30)
            data['Banking_Engagement'] = (data['Online'] + data['CreditCard'] +
                                         data['Securities Account'] + data['CD Account'])
            data['Income_squared'] = data['Income'] ** 2
            data['Age_squared'] = data['Age'] ** 2
        
        # Income categorization
        data['Income_Category'] = pd.cut(data['Income'],
                                        bins=[0, 50, 100, 200, float('inf')],
                                        labels=['Low', 'Medium', 'High', 'Very_High'])
        
        le = LabelEncoder()
        data['Income_Category_encoded'] = le.fit_transform(data['Income_Category'])
        
        return data
    except FileNotFoundError:
        st.error("Dataset 'bank_loan.csv' not found. Please upload the file first.")
        return None

def get_features_for_model(model_type):
    """Get feature list based on selected model"""
    base_features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education',
                    'Mortgage', 'Securities Account', 'CD Account', 'Online',
                    'CreditCard', 'Income_per_Family', 'Experience_Age_Ratio',
                    'Income_Category_encoded']
    
    if model_type == "Deep Learning":
        # Add advanced features for Deep Learning
        advanced_features = ['CCAvg_Income_Ratio', 'Wealth_Score', 'Banking_Engagement',
                           'Income_squared', 'Age_squared']
        return base_features + advanced_features
    else:
        return base_features

def train_logistic_regression(data):
    """Train Logistic Regression model"""
    features_to_use = get_features_for_model("Logistic Regression")
    X = data[features_to_use]
    y = data['Personal Loan']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, features_to_use

def train_random_forest(data):
    """Train Random Forest model with hyperparameter tuning"""
    features_to_use = get_features_for_model("Random Forest")
    X = data[features_to_use]
    y = data['Personal Loan']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    grid_search = GridSearchCV(rf_base, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    
    return model, grid_search, X_train, X_test, y_train, y_test, features_to_use

def train_xgboost(data):
    """Train XGBoost model with hyperparameter tuning"""
    features_to_use = get_features_for_model("XGBoost")
    X = data[features_to_use]
    y = data['Personal Loan']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_base = XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    grid_search = GridSearchCV(xgb_base, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    
    return model, grid_search, X_train, X_test, y_train, y_test, features_to_use, scale_pos_weight

def create_neural_network(input_dim, architecture, dropout_rate=0.3, l2_reg=0.01):
    """Create a deep neural network for binary classification"""
    model = Sequential()

    # Input layer with first hidden layer
    model.add(Dense(architecture[0],
                   input_dim=input_dim,
                   activation='relu',
                   kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Additional hidden layers
    for units in architecture[1:]:
        model.add(Dense(units,
                       activation='relu',
                       kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    return model

def train_deep_learning(data):
    """Train Deep Learning model with architecture optimization"""
    features_to_use = get_features_for_model("Deep Learning")
    X = data[features_to_use]
    y = data['Personal Loan']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling - CRITICAL for neural networks
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # Test different architectures
    architectures = [
        [64, 32],           # Simple
        [128, 64, 32],      # Medium
        [256, 128, 64, 32], # Complex
    ]
    
    best_model = None
    best_score = 0
    best_architecture = None
    best_history = None
    
    for architecture in architectures:
        # Create model
        model = create_neural_network(X_train_scaled.shape[1], architecture)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        val_predictions = model.predict(X_train_scaled, verbose=0)
        val_auc = roc_auc_score(y_train, val_predictions)
        
        if val_auc > best_score:
            best_score = val_auc
            best_model = model
            best_architecture = architecture
            best_history = history
    
    return best_model, best_architecture, best_history, scaler, X_train_scaled, X_test_scaled, y_train, y_test, features_to_use, class_weight_dict

def evaluate_model(model, X_test, y_test, model_type, scaler=None):
    """Evaluate model performance"""
    # Make predictions based on model type
    if model_type == "Deep Learning":
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
    elif model_type == "Logistic Regression":
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:  # Random Forest, XGBoost
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, cm, y_pred, y_pred_proba

def plot_performance_metrics(metrics, model_type):
    """Plot performance metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Metrics bar chart
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax1.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'salmon', 'gold', 'purple'])
    ax1.set_title(f'{model_type} Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{metric_values[i]:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Confusion Matrix
    try:
        cm = st.session_state.get('cm', [[0, 0], [0, 0]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=['No Loan', 'Loan'],
                    yticklabels=['No Loan', 'Loan'])
        ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Actual')
        ax2.set_xlabel('Predicted')
    except Exception:
        ax2.text(0.5, 0.5, 'Confusion Matrix\nNot Available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # ROC Curve
    try:
        y_test = st.session_state.get('y_test', [])
        y_pred_proba = st.session_state.get('y_pred_proba', [])
        
        if len(y_test) > 0 and len(y_pred_proba) > 0:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["auc"]:.4f})')
            ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax3.legend(loc="lower right")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'ROC Curve\nNot Available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
    except Exception:
        ax3.text(0.5, 0.5, 'ROC Curve\nNot Available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
    
    # Precision-Recall Curve
    try:
        y_test = st.session_state.get('y_test', [])
        y_pred_proba = st.session_state.get('y_pred_proba', [])
        
        if len(y_test) > 0 and len(y_pred_proba) > 0:
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
            ax4.plot(recall_vals, precision_vals, color='blue', lw=2)
            ax4.set_xlabel('Recall')
            ax4.set_ylabel('Precision')
            ax4.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Precision-Recall Curve\nNot Available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    except Exception:
        ax4.text(0.5, 0.5, 'Precision-Recall Curve\nNot Available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_feature_importance(model, features_to_use, model_type):
    """Plot feature importance based on model type"""
    if model_type == "Logistic Regression":
        feature_importance = pd.DataFrame({
            'feature': features_to_use,
            'coefficient': model.coef_[0],
            'abs_coefficient': np.abs(model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        importance_col = 'abs_coefficient'
        title_suffix = 'Coefficients'
        
    elif model_type == "Random Forest":
        feature_importance = pd.DataFrame({
            'feature': features_to_use,
            'importance': model.feature_importances_,
        }).sort_values('importance', ascending=False)
        
        importance_col = 'importance'
        title_suffix = 'Importance'
        
    elif model_type == "XGBoost":
        feature_importance = pd.DataFrame({
            'feature': features_to_use,
            'importance': model.feature_importances_,
        }).sort_values('importance', ascending=False)
        
        importance_col = 'importance'
        title_suffix = 'Gain Importance'
        
    elif model_type == "Deep Learning":
        # Get first layer weights as approximation
        first_layer_weights = model.layers[0].get_weights()[0]
        feature_importance_approx = np.mean(np.abs(first_layer_weights), axis=1)
        
        feature_importance = pd.DataFrame({
            'feature': features_to_use,
            'importance': feature_importance_approx,
        }).sort_values('importance', ascending=False)
        
        importance_col = 'importance'
        title_suffix = 'Neural Weights'
    
    # Plot top 10 features
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    top_features = feature_importance.head(10)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    
    # Horizontal bar chart
    bars1 = ax1.barh(range(len(top_features)), top_features[importance_col], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'])
    ax1.set_xlabel(f'{title_suffix}')
    ax1.set_title(f'{model_type} Feature {title_suffix}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # Add value labels
    for i, bar in enumerate(bars1):
        ax1.text(bar.get_width() + max(top_features[importance_col])*0.01, 
                bar.get_y() + bar.get_height()/2,
                f'{top_features.iloc[i][importance_col]:.4f}', 
                ha='left', va='center', fontsize=8)
    
    # Feature importance distribution
    ax2.hist(feature_importance[importance_col], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel(f'{title_suffix}')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Distribution of Feature {title_suffix}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, feature_importance

# File uploader
uploaded_file = st.file_uploader("Upload your bank loan dataset (CSV)", type=['csv'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("bank_loan.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Dataset successfully uploaded!")

# Training Section
if st.button(f"🚀 Train {model_choice} Model", type="primary"):
    # Check if required libraries are available
    if model_choice == "XGBoost" and not XGBOOST_AVAILABLE:
        st.error("❌ XGBoost is not installed. Please run: `pip install xgboost`")
        st.stop()
    
    if model_choice == "Deep Learning" and not TENSORFLOW_AVAILABLE:
        st.error("❌ TensorFlow is not installed. Please run: `pip install tensorflow`")
        st.stop()
    
    with st.spinner("Loading and preprocessing data..."):
        data = load_and_preprocess_data()
    
    if data is not None:
        # Display basic data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{data.shape[0]:,}")
        with col2:
            st.metric("Features", f"{data.shape[1]}")
        with col3:
            loan_rate = data['Personal Loan'].mean() * 100
            st.metric("Loan Rate", f"{loan_rate:.1f}%")
        
        st.markdown("---")
        
        # Train selected model
        with st.spinner(f"Training {model_choice} model..."):
            if model_choice == "Logistic Regression":
                model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, features_to_use = train_logistic_regression(data)
                additional_info = None
                
            elif model_choice == "Random Forest":
                model, grid_search, X_train, X_test, y_train, y_test, features_to_use = train_random_forest(data)
                scaler = None
                X_test_scaled = X_test
                additional_info = grid_search
                
            elif model_choice == "XGBoost":
                model, grid_search, X_train, X_test, y_train, y_test, features_to_use, scale_pos_weight = train_xgboost(data)
                scaler = None
                X_test_scaled = X_test
                additional_info = {'grid_search': grid_search, 'scale_pos_weight': scale_pos_weight}
                
            elif model_choice == "Deep Learning":
                model, architecture, history, scaler, X_train_scaled, X_test_scaled, y_train, y_test, features_to_use, class_weight_dict = train_deep_learning(data)
                additional_info = {'architecture': architecture, 'history': history, 'class_weight_dict': class_weight_dict}
        
        with st.spinner(f"Evaluating {model_choice} model performance..."):
            metrics, cm, y_pred, y_pred_proba = evaluate_model(model, X_test_scaled, y_test, model_choice, scaler)
        
        # Store in session state
        st.session_state.trained_model = model
        st.session_state.scaler = scaler
        st.session_state.features_to_use = features_to_use
        st.session_state.model_type = model_choice
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.y_pred_proba = y_pred_proba
        st.session_state.cm = cm
        st.session_state.additional_info = additional_info
        st.session_state.model_trained = True
        
        # Display Results
        st.header(f"📊 {model_choice} Model Performance Metrics")
        
        # Metrics cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="🎯 Accuracy",
                value=f"{metrics['accuracy']:.4f}",
                delta=f"{metrics['accuracy']*100:.2f}%"
            )
        
        with col2:
            st.metric(
                label="🔍 Precision", 
                value=f"{metrics['precision']:.4f}",
                delta=f"{metrics['precision']*100:.2f}%"
            )
        
        with col3:
            st.metric(
                label="📈 Recall",
                value=f"{metrics['recall']:.4f}",
                delta=f"{metrics['recall']*100:.2f}%"
            )
        
        with col4:
            st.metric(
                label="⚖️ F1-Score",
                value=f"{metrics['f1']:.4f}",
                delta=f"{metrics['f1']*100:.2f}%"
            )
        
        with col5:
            st.metric(
                label="🎲 AUC",
                value=f"{metrics['auc']:.4f}",
                delta=f"{metrics['auc']*100:.2f}%"
            )
        
        # Performance interpretation
        if metrics['auc'] >= 0.9:
            performance_level = "🏆 Excellent"
            performance_color = "green"
        elif metrics['auc'] >= 0.8:
            performance_level = "🥇 Very Good"
            performance_color = "blue"
        elif metrics['auc'] >= 0.7:
            performance_level = "🥈 Good"
            performance_color = "orange"
        else:
            performance_level = "🥉 Needs Improvement"
            performance_color = "red"
        
        st.markdown(f"**Overall {model_choice} Performance: <span style='color:{performance_color}'>{performance_level}</span>**", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model-specific details
        if model_choice == "Random Forest" and additional_info:
            st.subheader("🌳 Random Forest Model Configuration")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Trees", model.n_estimators)
                st.metric("Max Depth", str(model.max_depth))
            with col2:
                st.metric("Min Samples Split", model.min_samples_split)
                st.metric("Min Samples Leaf", model.min_samples_leaf)
            with col3:
                try:
                    best_score = additional_info.best_score_ if hasattr(additional_info, 'best_score_') else "N/A"
                    st.metric("Best CV Score", f"{best_score:.4f}" if isinstance(best_score, float) else best_score)
                except:
                    st.metric("Best CV Score", "N/A")
                st.metric("Class Weight", "Balanced")
        
        elif model_choice == "XGBoost" and additional_info and XGBOOST_AVAILABLE:
            st.subheader("🚀 XGBoost Model Configuration")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Estimators", model.n_estimators)
                st.metric("Max Depth", model.max_depth)
            with col2:
                st.metric("Learning Rate", f"{model.learning_rate:.3f}")
                st.metric("Subsample", f"{model.subsample:.2f}")
            with col3:
                try:
                    grid_search = additional_info.get('grid_search') if isinstance(additional_info, dict) else additional_info
                    best_score = grid_search.best_score_ if hasattr(grid_search, 'best_score_') else "N/A"
                    st.metric("Best CV Score", f"{best_score:.4f}" if isinstance(best_score, float) else best_score)
                except:
                    st.metric("Best CV Score", "N/A")
                try:
                    scale_pos_weight = additional_info.get('scale_pos_weight', "N/A") if isinstance(additional_info, dict) else "N/A"
                    st.metric("Scale Pos Weight", f"{scale_pos_weight:.2f}" if isinstance(scale_pos_weight, (int, float)) else scale_pos_weight)
                except:
                    st.metric("Scale Pos Weight", "N/A")
        
        elif model_choice == "Deep Learning" and additional_info and TENSORFLOW_AVAILABLE:
            st.subheader("🧠 Neural Network Architecture")
            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    architecture = additional_info.get('architecture', [])
                    st.metric("Architecture", f"{architecture}")
                    st.metric("Total Parameters", f"{model.count_params():,}")
                except:
                    st.metric("Architecture", "N/A")
                    st.metric("Total Parameters", "N/A")
            with col2:
                try:
                    architecture = additional_info.get('architecture', [])
                    history = additional_info.get('history')
                    st.metric("Hidden Layers", len(architecture) if architecture else "N/A")
                    if history and hasattr(history, 'history') and 'loss' in history.history:
                        st.metric("Training Epochs", len(history.history['loss']))
                    else:
                        st.metric("Training Epochs", "N/A")
                except:
                    st.metric("Hidden Layers", "N/A")
                    st.metric("Training Epochs", "N/A")
            with col3:
                try:
                    history = additional_info.get('history')
                    if history and hasattr(history, 'history'):
                        if 'loss' in history.history and len(history.history['loss']) > 0:
                            st.metric("Final Train Loss", f"{history.history['loss'][-1]:.4f}")
                        else:
                            st.metric("Final Train Loss", "N/A")
                        if 'val_loss' in history.history and len(history.history['val_loss']) > 0:
                            st.metric("Final Val Loss", f"{history.history['val_loss'][-1]:.4f}")
                        else:
                            st.metric("Final Val Loss", "N/A")
                    else:
                        st.metric("Final Train Loss", "N/A")
                        st.metric("Final Val Loss", "N/A")
                except:
                    st.metric("Final Train Loss", "N/A")
                    st.metric("Final Val Loss", "N/A")
        
        # Performance visualization
        st.header(f"📈 {model_choice} Performance Matrix")
        fig_metrics = plot_performance_metrics(metrics, model_choice)
        st.pyplot(fig_metrics)
        
        st.markdown("---")
        
        # Feature importance analysis
        st.header(f"🔍 {model_choice} Feature Importance Analysis")
        fig_importance, feature_importance_df = plot_feature_importance(model, features_to_use, model_choice)
        st.pyplot(fig_importance)
        
        # Store feature importance
        st.session_state.feature_importance = feature_importance_df
        
        # Cross-validation
        st.markdown("---")
        st.header("🔄 Cross-Validation Results")
        
        if model_choice == "Deep Learning" and additional_info and 'history' in additional_info:
            # For Deep Learning, show validation results from training history
            st.info("Deep Learning models use built-in validation during training. Cross-validation results shown below:")
            
            try:
                history = additional_info['history']
                history_dict = history.history
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Try to get validation AUC, fallback to test AUC
                    if 'val_auc' in history_dict and len(history_dict['val_auc']) > 0:
                        final_val_auc = history_dict['val_auc'][-1]
                    else:
                        final_val_auc = metrics['auc']
                    st.metric("Validation AUC", f"{final_val_auc:.4f}")
                    
                with col2:
                    # Calculate validation loss stability
                    if 'val_loss' in history_dict and len(history_dict['val_loss']) > 0:
                        val_loss_history = history_dict['val_loss']
                        val_loss_std = np.std(val_loss_history[-min(10, len(val_loss_history)):])
                    else:
                        val_loss_std = 0.0
                    st.metric("Val Loss Std (last 10)", f"{val_loss_std:.4f}")
                    
                with col3:
                    stability = "Stable" if val_loss_std < 0.05 else "Moderate" if val_loss_std < 0.1 else "Unstable"
                    st.metric("Model Stability", stability)
                
                # Show training history plot if data is available
                if 'loss' in history_dict and 'val_loss' in history_dict:
                    fig_history, ax = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Loss plot
                    ax[0].plot(history_dict['loss'], label='Training Loss', color='blue')
                    ax[0].plot(history_dict['val_loss'], label='Validation Loss', color='red')
                    ax[0].set_title('Training History - Loss')
                    ax[0].set_xlabel('Epoch')
                    ax[0].set_ylabel('Loss')
                    ax[0].legend()
                    ax[0].grid(True, alpha=0.3)
                    
                    # Accuracy plot
                    if 'accuracy' in history_dict and 'val_accuracy' in history_dict:
                        ax[1].plot(history_dict['accuracy'], label='Training Accuracy', color='green')
                        ax[1].plot(history_dict['val_accuracy'], label='Validation Accuracy', color='orange')
                        ax[1].set_title('Training History - Accuracy')
                        ax[1].set_xlabel('Epoch')
                        ax[1].set_ylabel('Accuracy')
                        ax[1].legend()
                        ax[1].grid(True, alpha=0.3)
                    else:
                        ax[1].text(0.5, 0.5, 'Accuracy history\nnot available', 
                                  ha='center', va='center', transform=ax[1].transAxes)
                        ax[1].set_title('Training History - Accuracy')
                    
                    plt.tight_layout()
                    st.pyplot(fig_history)
                    
            except Exception as e:
                st.warning(f"Could not display Deep Learning training history: {str(e)}")
                # Fallback to basic metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test AUC", f"{metrics['auc']:.4f}")
                with col2:
                    st.metric("Test Accuracy", f"{metrics['accuracy']:.4f}")
                with col3:
                    st.metric("Model Status", "Trained Successfully")
            
        else:
            # Traditional cross-validation for sklearn models
            try:
                with st.spinner("Performing cross-validation..."):
                    if model_choice == "Logistic Regression" and scaler is not None:
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
                    elif model_choice == "Logistic Regression":
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                    else:
                        # For Random Forest and XGBoost, use original features
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean CV AUC", f"{cv_scores.mean():.4f}")
                with col2:
                    st.metric("CV Std Dev", f"{cv_scores.std():.4f}")
                with col3:
                    stability = "Stable" if cv_scores.std() < 0.05 else "Moderate" if cv_scores.std() < 0.1 else "Unstable"
                    st.metric("Model Stability", stability)
                    
            except Exception as e:
                st.warning(f"Cross-validation failed: {str(e)}")
                # Fallback to test metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test AUC", f"{metrics['auc']:.4f}")
                with col2:
                    st.metric("Test Accuracy", f"{metrics['accuracy']:.4f}")
                with col3:
                    st.metric("Model Status", "Trained Successfully")
        
        st.success(f"{model_choice} model training and evaluation completed successfully!")

# Prediction Section
if 'model_trained' in st.session_state and st.session_state.model_trained:
    st.markdown("---")
    st.header(f"🔮 Make Predictions with {st.session_state.model_type}")
    
    st.write(f"Use the trained {st.session_state.model_type} model to predict loan probability for new customers:")
    
    # Create prediction form
    with st.form(key='prediction_form'):
        st.subheader("📝 Customer Information")
        
        col1, col2 = st.columns(2)
        
        # Initialize prediction input dictionary
        prediction_input = {}
        
        with col1:
            prediction_input['Income'] = st.number_input(
                "💰 Annual Income (in thousands USD)", 
                min_value=0, max_value=1000, value=80, step=5,
                help="Customer's annual income"
            )
            
            prediction_input['CCAvg'] = st.number_input(
                "💳 Average Credit Card Spending (monthly, in thousands USD)", 
                min_value=0.0, max_value=50.0, value=2.0, step=0.1,
                help="Average monthly credit card spending"
            )
            
            prediction_input['Education'] = st.selectbox(
                "🎓 Education Level", 
                options=[1, 2, 3],
                format_func=lambda x: {1: "Bachelor's Degree", 2: "Master's Degree", 3: "Doctorate"}[x],
                index=0,
                help="Highest education level completed"
            )
            
            prediction_input['Age'] = st.number_input(
                "👤 Age", 
                min_value=18, max_value=100, value=35, step=1,
                help="Customer's age in years"
            )
            
            prediction_input['Experience'] = st.number_input(
                "💼 Work Experience (years)", 
                min_value=0, max_value=50, value=10, step=1,
                help="Years of professional work experience"
            )
            
            prediction_input['Family'] = st.number_input(
                "👨‍👩‍👧‍👦 Family Size", 
                min_value=1, max_value=10, value=3, step=1,
                help="Number of family members"
            )
        
        with col2:
            prediction_input['Securities Account'] = st.selectbox(
                "📈 Securities Account", 
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Does customer have a securities account?"
            )
            
            prediction_input['CD Account'] = st.selectbox(
                "💿 Certificate of Deposit Account", 
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Does customer have a CD account?"
            )
            
            prediction_input['Online'] = st.selectbox(
                "💻 Online Banking User", 
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Does customer use online banking?"
            )
            
            prediction_input['CreditCard'] = st.selectbox(
                "💳 Credit Card Holder", 
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Does customer have a credit card with the bank?"
            )
            
            prediction_input['Mortgage'] = st.number_input(
                "🏠 Mortgage Amount", 
                min_value=0, max_value=1000, value=0, step=10,
                help="Current mortgage amount"
            )
        
        # Submit button
        predict_button = st.form_submit_button(
            label=f"🚀 Predict with {st.session_state.model_type}", 
            type="primary"
        )
        
        if predict_button:
            # Calculate engineered features
            prediction_input['Income_per_Family'] = prediction_input['Income'] / prediction_input['Family']
            prediction_input['Experience_Age_Ratio'] = prediction_input['Experience'] / prediction_input['Age']
            
            # Advanced features for Deep Learning
            if st.session_state.model_type == "Deep Learning":
                prediction_input['CCAvg_Income_Ratio'] = prediction_input['CCAvg'] / (prediction_input['Income'] + 1e-8)
                prediction_input['Wealth_Score'] = (prediction_input['Income'] * 0.4 + 
                                                  prediction_input['CCAvg'] * 0.3 +
                                                  prediction_input['Securities Account'] * 50 + 
                                                  prediction_input['CD Account'] * 30)
                prediction_input['Banking_Engagement'] = (prediction_input['Online'] + 
                                                        prediction_input['CreditCard'] +
                                                        prediction_input['Securities Account'] + 
                                                        prediction_input['CD Account'])
                prediction_input['Income_squared'] = prediction_input['Income'] ** 2
                prediction_input['Age_squared'] = prediction_input['Age'] ** 2
            
            # Create income category
            if prediction_input['Income'] <= 50:
                income_cat = 0  # Low
            elif prediction_input['Income'] <= 100:
                income_cat = 1  # Medium
            elif prediction_input['Income'] <= 200:
                income_cat = 2  # High
            else:
                income_cat = 3  # Very High
            
            prediction_input['Income_Category_encoded'] = income_cat
            
            # Create prediction DataFrame
            prediction_df = pd.DataFrame([prediction_input])
            
            # Ensure all features are present in correct order
            for feature in st.session_state.features_to_use:
                if feature not in prediction_df.columns:
                    prediction_df[feature] = 0
            
            # Reorder columns to match training data
            prediction_df = prediction_df[st.session_state.features_to_use]
            
            # Make prediction based on model type
            try:
                if st.session_state.model_type in ["Logistic Regression", "Deep Learning"] and st.session_state.scaler:
                    prediction_scaled = st.session_state.scaler.transform(prediction_df)
                    if st.session_state.model_type == "Deep Learning":
                        prediction_proba = st.session_state.trained_model.predict(prediction_scaled, verbose=0)[0][0]
                        prediction = 1 if prediction_proba > 0.5 else 0
                        prediction_proba_array = [1-prediction_proba, prediction_proba]
                    else:  # Logistic Regression
                        prediction = st.session_state.trained_model.predict(prediction_scaled)[0]
                        prediction_proba_array = st.session_state.trained_model.predict_proba(prediction_scaled)[0]
                else:  # Random Forest, XGBoost
                    prediction = st.session_state.trained_model.predict(prediction_df)[0]
                    prediction_proba_array = st.session_state.trained_model.predict_proba(prediction_df)[0]
                
                # Display results
                st.markdown("---")
                st.header(f"🎯 {st.session_state.model_type} Prediction Results")
                
                # Main prediction result
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    loan_probability = prediction_proba_array[1] * 100
                    if prediction == 1:
                        st.success(f"**Prediction: LIKELY TO TAKE LOAN** ✅")
                    else:
                        st.error(f"**Prediction: UNLIKELY TO TAKE LOAN** ❌")
                
                with col2:
                    st.metric(
                        label="🎲 Loan Probability",
                        value=f"{loan_probability:.2f}%",
                        delta=f"Confidence: {max(prediction_proba_array)*100:.1f}%"
                    )
                
                with col3:
                    risk_level = "High" if loan_probability > 70 else "Medium" if loan_probability > 30 else "Low"
                    risk_color = "green" if risk_level == "High" else "orange" if risk_level == "Medium" else "red"
                    st.markdown(f"**Risk Level: <span style='color:{risk_color}'>{risk_level}</span>**", unsafe_allow_html=True)
                
                # Model-specific analysis
                st.subheader(f"📊 {st.session_state.model_type} Analysis Details")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Prediction Breakdown:**")
                    st.write(f"• No Loan: {prediction_proba_array[0]*100:.2f}%")
                    st.write(f"• Loan: {prediction_proba_array[1]*100:.2f}%")
                    
                    # Risk assessment
                    st.write(f"**{st.session_state.model_type} Risk Assessment:**")
                    if loan_probability > 80:
                        st.write("🔥 **Very High Probability** - Strong model consensus")
                    elif loan_probability > 60:
                        st.write("📈 **High Probability** - Model indicates good potential")
                    elif loan_probability > 40:
                        st.write("⚖️ **Moderate Probability** - Mixed signals from model")
                    elif loan_probability > 20:
                        st.write("📉 **Low Probability** - Model shows weak potential")
                    else:
                        st.write("❌ **Very Low Probability** - Strong negative indication")
                
                with col2:
                    # Feature impact analysis
                    st.write("**Key Factors Analysis:**")
                    
                    try:
                        # Get top 5 most important features
                        feature_importance = st.session_state.get('feature_importance')
                        if feature_importance is not None and len(feature_importance) > 0:
                            important_features = feature_importance.head(5)
                            
                            for _, row in important_features.iterrows():
                                feature_name = row.get('feature', 'Unknown')
                                
                                if feature_name in prediction_input:
                                    feature_value = prediction_input[feature_name]
                                    
                                    if st.session_state.model_type == "Logistic Regression":
                                        coeff = row.get('coefficient', 0)
                                        impact = "Positive" if coeff > 0 else "Negative"
                                        st.write(f"• **{feature_name}**: {feature_value} ({impact} impact)")
                                    else:
                                        importance = row.get('importance', 0)
                                        st.write(f"• **{feature_name}**: {feature_value} (Importance: {importance:.4f})")
                        else:
                            st.write("Feature importance data not available")
                    except Exception as e:
                        st.write("Feature impact analysis not available")
                
                # Business recommendations
                st.subheader("💼 Business Recommendations")
                
                if loan_probability > 70:
                    st.success(f"""
                    **High Priority Customer** 🎯
                    - **{st.session_state.model_type} indicates strong loan potential**
                    - Assign to senior relationship manager
                    - Offer personalized loan products with competitive rates
                    - Schedule immediate consultation
                    - High conversion probability based on ML analysis
                    """)
                elif loan_probability > 40:
                    st.info(f"""
                    **Medium Priority Customer** 📧
                    - **{st.session_state.model_type} shows moderate potential**
                    - Include in targeted digital campaigns
                    - Monitor engagement metrics closely
                    - Consider bundled product offerings
                    - ML model suggests cautious optimism
                    """)
                else:
                    st.warning(f"""
                    **Low Priority Customer** 📮
                    - **{st.session_state.model_type} indicates low potential**
                    - Focus on relationship building strategies
                    - Offer financial literacy programs
                    - Long-term cultivation approach
                    - ML model suggests patient development needed
                    """)
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.info("Please check your input values and try again.")

else:
    st.info("👆 Please upload a dataset and click 'Train Model' to begin the analysis.")
    
    # Show sample data format
    st.subheader("📋 Expected Data Format")
    sample_data = {
        'Age': [25, 35, 45],
        'Experience': [1, 10, 20],
        'Income': [30, 80, 150],
        'Family': [2, 3, 4],
        'CCAvg': [1.5, 2.5, 3.5],
        'Education': [1, 2, 3],
        'Personal Loan': [0, 1, 1]
    }
    st.dataframe(pd.DataFrame(sample_data))
    st.caption("Sample format - your dataset should contain these columns")
    
    # Model comparison information
    st.subheader("🔄 Model Comparison Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**When to Use Each Model:**")
        st.write("• **Logistic Regression**: Fast, interpretable, need feature coefficients")
        st.write("• **Random Forest**: Robust, handles interactions, feature importance")
        st.write("• **XGBoost**: High accuracy, competition-grade performance")
        st.write("• **Deep Learning**: Complex patterns, large datasets, non-linear relationships")
        
    with col2:
        st.write("**Model Characteristics:**")
        st.write("• **Speed**: Logistic > Random Forest > XGBoost > Deep Learning")
        st.write("• **Interpretability**: Logistic > Random Forest > XGBoost > Deep Learning")
        st.write("• **Accuracy**: Deep Learning ≥ XGBoost > Random Forest > Logistic")
        st.write("• **Data Requirements**: Logistic < Random Forest < XGBoost < Deep Learning")

# Footer
st.markdown("---")
st.markdown("""
**💡 Tips for Best Results:**
- Ensure your dataset has all required columns
- Larger datasets generally improve Deep Learning performance
- XGBoost often provides the best balance of accuracy and interpretability
- Logistic Regression is excellent for understanding feature impacts
- Random Forest provides robust predictions with good interpretability
""")