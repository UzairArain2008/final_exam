# TASK E: MODEL DEPLOYMENT USING STREAMLIT (5 Marks)
# Filename: app.py
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1e40af;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
    }
    .survived {
        background-color: #d1fae5;
        color: #065f46;
        border: 2px solid #10b981;
    }
    .not-survived {
        background-color: #fee2e2;
        color: #991b1b;
        border: 2px solid #ef4444;
    }
    .stButton>button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    </style>
""", unsafe_allow_html=True)

# Load the saved model and preprocessing objects
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        pca = joblib.load('pca.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        # Try to load the model name
        try:
            model_name = joblib.load('best_model_name.pkl')
        except:
            model_name = "Machine Learning Model"
        
        return model, scaler, pca, feature_names, model_name
    except FileNotFoundError as e:
        st.error(f"⚠️ Model file not found: {e}")
        st.info("Please make sure you have run 'titanic_classification.py' first to generate the model files.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None, None

# Load models
model, scaler, pca, feature_names, model_name = load_models()

# Header
st.markdown('<h1 class="main-header">🚢 Titanic Survival Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predict passenger survival using Machine Learning")

if model_name:
    st.markdown(f"**Model:** {model_name}")

st.markdown("---")

if model is not None:
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Passenger Information")
        
        # Passenger Class
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            index=2,  # Default to 3rd class
            help="1 = First Class, 2 = Second Class, 3 = Third Class"
        )
        
        # Sex
        sex = st.radio(
            "Sex",
            options=["male", "female"],
            horizontal=True,
            help="Gender of the passenger"
        )
        
        # Age
        age = st.slider(
            "Age",
            min_value=0,
            max_value=80,
            value=30,
            help="Age in years"
        )
        
        # Number of Siblings/Spouses
        sibsp = st.number_input(
            "Number of Siblings/Spouses Aboard",
            min_value=0,
            max_value=8,
            value=0,
            help="Number of siblings or spouses traveling with the passenger"
        )
    
    with col2:
        st.subheader("🎫 Ticket Information")
        
        # Number of Parents/Children
        parch = st.number_input(
            "Number of Parents/Children Aboard",
            min_value=0,
            max_value=6,
            value=0,
            help="Number of parents or children traveling with the passenger"
        )
        
        # Fare
        fare = st.number_input(
            "Fare (in dollars)",
            min_value=0.0,
            max_value=600.0,
            value=32.0,
            step=0.5,
            help="Passenger fare in dollars"
        )
        
        # Port of Embarkation
        embarked = st.selectbox(
            "Port of Embarkation",
            options=["C", "Q", "S"],
            index=2,  # Default to Southampton
            format_func=lambda x: {
                "C": "Cherbourg (C)", 
                "Q": "Queenstown (Q)", 
                "S": "Southampton (S)"
            }[x],
            help="Port where passenger embarked"
        )
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Survival", type="primary"):
        try:
            # Create input dataframe matching the training data structure
            # Features order: pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S, pclass_2, pclass_3
            
            input_dict = {
                'pclass': pclass,
                'age': age,
                'sibsp': sibsp,
                'parch': parch,
                'fare': fare,
                'sex_male': 1 if sex == 'male' else 0,
                'embarked_Q': 1 if embarked == 'Q' else 0,
                'embarked_S': 1 if embarked == 'S' else 0,
                'pclass_2': 1 if pclass == 2 else 0,
                'pclass_3': 1 if pclass == 3 else 0
            }
            
            # Create DataFrame with features in correct order
            input_data = pd.DataFrame([input_dict])
            
            # Reorder columns to match feature_names
            input_data = input_data[feature_names]
            
            # Debug info (can be removed in production)
            with st.expander("🔍 Debug Information"):
                st.write("Input Data Shape:", input_data.shape)
                st.write("Expected Features:", feature_names)
                st.write("Input Features:", list(input_data.columns))
                st.write("Input Values:")
                st.dataframe(input_data)
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Apply PCA transformation
            input_pca = pca.transform(input_scaled)
            
            # Make prediction
            prediction = model.predict(input_pca)[0]
            
            # Get probability if available
            prediction_proba = None
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(input_pca)[0]
            
            # Display results
            st.markdown("### 📊 Prediction Results")
            
            if prediction == 1:
                st.markdown(
                    '<div class="prediction-box survived">✅ SURVIVED<br>The passenger would likely have survived the Titanic disaster</div>',
                    unsafe_allow_html=True
                )
                st.balloons()
            else:
                st.markdown(
                    '<div class="prediction-box not-survived">❌ NOT SURVIVED<br>The passenger would likely not have survived the Titanic disaster</div>',
                    unsafe_allow_html=True
                )
            
            # Show probability if available
            if prediction_proba is not None:
                st.markdown("### 🎯 Prediction Confidence")
                col_prob1, col_prob2 = st.columns(2)
                
                with col_prob1:
                    st.metric(
                        "Probability of Not Surviving",
                        f"{prediction_proba[0]:.2%}",
                        delta=None
                    )
                
                with col_prob2:
                    st.metric(
                        "Probability of Surviving",
                        f"{prediction_proba[1]:.2%}",
                        delta=None
                    )
                
                # Progress bars for visualization
                st.write("")
                st.write("**Confidence Visualization:**")
                st.progress(prediction_proba[0], text=f"Not Survived: {prediction_proba[0]:.1%}")
                st.progress(prediction_proba[1], text=f"Survived: {prediction_proba[1]:.1%}")
            
            # Show input summary
            with st.expander("📋 Input Summary"):
                summary_data = {
                    "Feature": [
                        "Passenger Class", 
                        "Sex", 
                        "Age", 
                        "Siblings/Spouses", 
                        "Parents/Children", 
                        "Fare", 
                        "Port of Embarkation"
                    ],
                    "Value": [
                        f"{pclass} ({'First' if pclass==1 else 'Second' if pclass==2 else 'Third'} Class)",
                        sex.capitalize(),
                        f"{age} years",
                        sibsp,
                        parch,
                        f"${fare:.2f}",
                        {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[embarked]
                    ]
                }
                st.table(pd.DataFrame(summary_data))
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
            st.exception(e)
            st.info("Please check that all inputs are valid and the model files are properly loaded.")
    
    # Add sample predictions
    st.markdown("---")
    st.subheader("💡 Try These Sample Passengers")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        if st.button("👨 3rd Class Male"):
            st.info("Try: 3rd Class, Male, Age 25, 0 family, $7.25 fare, Southampton")
    
    with col_s2:
        if st.button("👩 1st Class Female"):
            st.info("Try: 1st Class, Female, Age 35, 1 family, $75 fare, Cherbourg")
    
    with col_s3:
        if st.button("👨‍👩‍👧 2nd Class Family"):
            st.info("Try: 2nd Class, Female, Age 28, 1 spouse, 2 children, $30 fare, Queenstown")

else:
    st.error("⚠️ Model files not found!")
    st.info("""
    ### Please follow these steps:
    
    1. **Run the training script first:**
       ```bash
       python titanic_classification.py
       ```
    
    2. **This will generate the required files:**
       - best_model.pkl
       - scaler.pkl
       - pca.pkl
       - feature_names.pkl
       - best_model_name.pkl
    
    3. **Then run this Streamlit app:**
       ```bash
       streamlit run app.py
       ```
    
    4. **Make sure all files are in the same directory as this app**
    """)

# Sidebar with information
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/300px-RMS_Titanic_3.jpg", 
             use_container_width=True)
    
    st.header("ℹ️ About This App")
    st.info("""
    This application uses a Machine Learning model trained on the Titanic dataset 
    to predict passenger survival probability.
    
    **Model Details:**
    - Dataset: Titanic passengers (891 records)
    - Features: PCA-transformed (95% variance)
    - Algorithms: Logistic Regression, Random Forest, SVM
    - Preprocessing: StandardScaler + PCA
    """)
    
    if model_name:
        st.success(f"**Active Model:** {model_name}")
    
    st.markdown("---")
    
    st.header("📚 Feature Guide")
    st.markdown("""
    **Passenger Class (Pclass):**
    - 1st Class: Upper class, expensive tickets
    - 2nd Class: Middle class
    - 3rd Class: Lower class, cheapest tickets
    
    **Sex:**
    - Male or Female
    
    **Age:**
    - Age in years (0-80)
    
    **SibSp:**
    - Number of siblings or spouses aboard
    
    **Parch:**
    - Number of parents or children aboard
    
    **Fare:**
    - Ticket price in dollars
    
    **Embarked:**
    - C = Cherbourg
    - Q = Queenstown
    - S = Southampton
    """)
    
    st.markdown("---")
    
    st.header("📊 Historical Context")
    st.markdown("""
    The RMS Titanic sank on April 15, 1912 after hitting an iceberg.
    
    **Survival Statistics:**
    - Total Passengers: 891
    - Survived: 342 (38.4%)
    - Not Survived: 549 (61.6%)
    
    **Key Factors:**
    - Women and children first policy
    - Class-based lifeboat access
    - Crew experience
    """)
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit and scikit-learn")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.9rem;'>"
    "Data Science Lab Exam - Titanic Survival Prediction | Powered by Streamlit & scikit-learn"
    "</p>",
    unsafe_allow_html=True
)