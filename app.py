import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
from scipy.stats import zscore

# Set page configuration
st.set_page_config(
    page_title="Marketing Campaign Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Define functions for data processing
def load_data():
    uploaded_file = st.file_uploader("Upload your Excel dataset", type=['xlsx'])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df
    return None

def clean_data(df):
    if df is not None:
        # Handle missing values
        df.dropna(subset=['Income', 'Year_Birth'], inplace=True)
        df['Education'].fillna(df['Education'].mode()[0], inplace=True)
        df['Marital_Status'].fillna(df['Marital_Status'].mode()[0], inplace=True)
        # Drop duplicates
        df.drop_duplicates(inplace=True)
        # Convert date column
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
        # Calculate age
        df['Age'] = 2023 - df['Year_Birth']
        # Create total spending column
        df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
        # Remove outliers using Z-score
        numerical_cols = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                        'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 
                        'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
        # Calculate Z-scores
        z_scores = df[numerical_cols].apply(zscore)
        df_cleaned = df[(abs(z_scores) <= 3).all(axis=1)]
        return df_cleaned
    return None

def train_model(df):
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)
    # Create features
    df_encoded['Engagement_Score'] = (
        df_encoded['AcceptedCmp1'] + 
        df_encoded['AcceptedCmp2'] + 
        df_encoded['AcceptedCmp3'] + 
        df_encoded['AcceptedCmp4'] + 
        df_encoded['AcceptedCmp5']
    )
    # Define target and features
    X = df_encoded.drop(['Response', 'Dt_Customer', 'ID'], axis=1, errors='ignore')
    X = X.select_dtypes(include=['int64', 'float64'])
    y = df_encoded['Response']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    # Prepare results
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:,1]
    # Feature importance
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    return model, scaler, X_test_scaled, y_test, y_pred, y_pred_proba, feature_importances, X.columns

def predict_response(model, scaler, feature_vector):
    # Scale the input features
    scaled_features = scaler.transform([feature_vector])
    # Predict
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]
    return prediction, probability

# Main app layout
st.title("Marketing Campaign Analysis Dashboard")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Campaign Analysis", "Channel Effectiveness", "Product Analysis", "Customer Demographics", "ROI Analysis", "Predictive Model"])

# Load data
data = load_data()
if data is not None:
    # Clean data
    clean_data_toggle = st.sidebar.checkbox("Clean Data", value=True)
    if clean_data_toggle:
        data = clean_data(data)
    
    # Display different pages based on selection
    if page == "Data Overview":
        st.header("Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Dataset Shape:", data.shape)
            st.write("Missing Values:", data.isnull().sum().sum())
        with col2:
            if st.checkbox("Show Data Types"):
                st.write(data.dtypes)
        if st.checkbox("Show Raw Data"):
            st.dataframe(data.head(10))
        st.subheader("Data Distribution")
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_col = st.selectbox("Select column to visualize", numeric_cols)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data[selected_col], kde=True, ax=ax)
        st.pyplot(fig)
    elif page == "Campaign Analysis":
        st.header("Campaign Analysis")
        # Calculate engagement rates
        campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
        engagement_rates = data[campaign_columns].mean() * 100
        st.subheader("Campaign Engagement Rates")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=engagement_rates.index, y=engagement_rates.values, ax=ax)
        plt.ylabel("Engagement Rate (%)")
        plt.title("Engagement Rates Across Campaigns")
        st.pyplot(fig)
        st.subheader("Response by Age Group")
        # Create age groups
        bins = [0, 30, 40, 50, 60, 100]
        labels = ['<30', '30-40', '40-50', '50-60', '>60']
        data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)
        conversion_by_age = data.groupby('Age_Group')['Response'].mean() * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=conversion_by_age.index, y=conversion_by_age.values, ax=ax)
        plt.ylabel("Conversion Rate (%)")
        plt.title("Conversion Rates by Age Group")
        st.pyplot(fig)
    elif page == "Channel Effectiveness":
        st.header("Channel Effectiveness")
        avg_web_visits = data['NumWebVisitsMonth'].mean()
        avg_web_purchases = data['NumWebPurchases'].mean()
        avg_catalog_purchases = data['NumCatalogPurchases'].mean()
        avg_store_purchases = data['NumStorePurchases'].mean()
        # Calculate conversion rates
        conversion_rate_web = (avg_web_purchases / avg_web_visits) * 100
        conversion_rate_catalog = (avg_catalog_purchases / avg_web_visits) * 100
        conversion_rate_store = (avg_store_purchases / avg_web_visits) * 100
        conversion_rates = pd.DataFrame({
            'Channel': ['Web', 'Catalog', 'Store'],
            'Conversion Rate (%)': [conversion_rate_web, conversion_rate_catalog, conversion_rate_store]
        })
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Conversion Rates by Channel")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=conversion_rates, x='Channel', y='Conversion Rate (%)', ax=ax)
            st.pyplot(fig)
        with col2:
            st.subheader("Channel Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.pie(conversion_rates['Conversion Rate (%)'], labels=conversion_rates['Channel'], autopct='%1.1f%%', startangle=90)
            st.pyplot(fig)
    elif page == "Product Analysis":
        st.header("Product Analysis")
        # Calculate total sales for each product category
        total_sales = {
            'Wines': data['MntWines'].sum(),
            'Fruits': data['MntFruits'].sum(),
            'Meat Products': data['MntMeatProducts'].sum(),
            'Fish Products': data['MntFishProducts'].sum(),
            'Sweet Products': data['MntSweetProducts'].sum(),
            'Gold Products': data['MntGoldProds'].sum()
        }
        # Convert to DataFrame
        total_sales_df = pd.DataFrame(list(total_sales.items()), columns=['Product', 'Total_Sales'])
        total_sales_df = total_sales_df.sort_values(by='Total_Sales', ascending=False)
        st.subheader("Total Sales by Product Category")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Total_Sales', y='Product', data=total_sales_df, ax=ax)
        st.pyplot(fig)
        # Product sales by demographic
        demographic_option = st.selectbox("Analyze Product Sales by:", ["Age Group", "Marital Status", "Education"])
        if demographic_option == "Age Group":
            # Calculate age and create groups if not done already
            if 'Age_Group' not in data.columns:
                bins = [0, 30, 40, 50, 60, 100]
                labels = ['<30', '30-40', '40-50', '50-60', '>60']
                data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)
            group_sales = data.groupby('Age_Group')[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum()
            st.subheader("Product Sales by Age Group")
        elif demographic_option == "Marital Status":
            group_sales = data.groupby('Marital_Status')[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum()
            st.subheader("Product Sales by Marital Status")
        else:  # Education
            group_sales = data.groupby('Education')[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                              'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum()
            st.subheader("Product Sales by Education Level")
        fig, ax = plt.subplots(figsize=(12, 8))
        group_sales.plot(kind='bar', stacked=True, ax=ax)
        plt.legend(title="Product Category")
        st.pyplot(fig)
    elif page == "Customer Demographics":
        st.header("Customer Demographics")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Age Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data['Age'], bins=30, kde=True, ax=ax)
            st.pyplot(fig)
        with col2:
            st.subheader("Income Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data['Income'], bins=30, kde=True, ax=ax)
            st.pyplot(fig)
        st.subheader("Education Level Distribution")
        education_counts = data['Education'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=education_counts.index, y=education_counts.values, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.subheader("Marital Status Distribution")
        marital_counts = data['Marital_Status'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=marital_counts.index, y=marital_counts.values, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    elif page == "ROI Analysis":
        st.header("ROI Analysis")
        # Calculate ROI metrics if columns exist
        if all(col in data.columns for col in ['Z_Revenue', 'Z_CostContact']):
            data['Net_Profit'] = data['Z_Revenue'] - data['Z_CostContact']
            data['ROI'] = (data['Net_Profit'] / data['Z_CostContact']) * 100
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Net Profit", f"${data['Net_Profit'].mean():.2f}")
            with col2:
                st.metric("Average ROI", f"{data['ROI'].mean():.2f}%")
            st.subheader("ROI Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data['ROI'], bins=30, kde=True, ax=ax)
            st.pyplot(fig)
            # ROI by customer segment
            if 'Age_Group' not in data.columns:
                bins = [0, 30, 40, 50, 60, 100]
                labels = ['<30', '30-40', '40-50', '50-60', '>60']
                data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)
            roi_by_age = data.groupby('Age_Group')['ROI'].mean()
            st.subheader("Average ROI by Age Group")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=roi_by_age.index, y=roi_by_age.values, ax=ax)
            plt.ylabel("Average ROI (%)")
            st.pyplot(fig)
        else:
            st.warning("Revenue or Cost columns not found in the dataset")
    elif page == "Predictive Model":
        st.header("Predictive Model Training")
        
        # Add explanations
        st.write("""
        This section allows you to train a machine learning model to predict customer response to campaigns.
        The model will analyze patterns in your data to identify factors that influence campaign success.
        """)
        
        # Store the model and scaler in session state so they persist between interactions
        if 'model' not in st.session_state:
            st.session_state.model = None
            st.session_state.scaler = None
            st.session_state.feature_names = None
        
        train_tab, predict_tab = st.tabs(["Train Model", "Make Predictions"])
        
        with train_tab:
            if st.button("Train Model"):
                with st.spinner("Training model... This may take a moment."):
                    try:
                        model, scaler, X_test_scaled, y_test, y_pred, y_pred_proba, feature_importances, feature_names = train_model(data)
                        
                        # Store in session state
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.feature_names = feature_names
                        
                        if model is not None:
                            st.success("Model trained successfully!")
                            
                            st.subheader("Model Performance")
                            st.text(classification_report(y_test, y_pred))
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Confusion Matrix")
                                cm = confusion_matrix(y_test, y_pred)
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                plt.ylabel('Actual')
                                plt.xlabel('Predicted')
                                st.pyplot(fig)
                            
                            with col2:
                                st.subheader("ROC Curve")
                                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                                auc = roc_auc_score(y_test, y_pred_proba)
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
                                plt.plot([0, 1], [0, 1], 'k--')
                                plt.xlabel('False Positive Rate')
                                plt.ylabel('True Positive Rate')
                                plt.legend(loc='lower right')
                                st.pyplot(fig)
                            
                            st.subheader("Feature Importance")
                            fig, ax = plt.subplots(figsize=(12, 8))
                            sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15), ax=ax)
                            st.pyplot(fig)
                            
                            # Save model option
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Download Model"):
                                    joblib.dump(model, 'marketing_campaign_model.pkl')
                                    joblib.dump(scaler, 'feature_scaler.pkl')
                                    st.success("Model saved as 'marketing_campaign_model.pkl'")
                    except Exception as e:
                        st.error(f"An error occurred during model training: {e}")
        
        with predict_tab:
            st.subheader("Prediction Interface")
            
            if st.session_state.model is None:
                st.warning("Please train a model first before making predictions")
            else:
                st.write("Enter customer information to predict campaign response likelihood:")
                
                # Create a form for predictions with the most important features
                # This is a simplified version - you may need to adjust based on your actual features
                with st.form("prediction_form"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        income = st.number_input("Income", min_value=0, value=50000)
                        age = st.number_input("Age", min_value=18, max_value=100, value=40)
                        total_spending = st.number_input("Total Spending", min_value=0, value=500)
                    
                    with col2:
                        web_visits = st.number_input("Web Visits per Month", min_value=0, value=5)
                        catalog_purchases = st.number_input("Catalog Purchases", min_value=0, value=2)
                        web_purchases = st.number_input("Web Purchases", min_value=0, value=3)
                    
                    with col3:
                        store_purchases = st.number_input("Store Purchases", min_value=0, value=3)
                        has_children = st.checkbox("Has Children")
                        previous_campaign = st.checkbox("Responded to Previous Campaign")
                    
                    # Submit button
                    submitted = st.form_submit_button("Predict Response")
                    
                    if submitted:
                        # Create a feature dictionary that matches the expected features
                        # This is simplified - in a real app, you'd need all features used in training
                        features = {
                            'Income': income,
                            'Age': age,
                            'NumWebVisitsMonth': web_visits,
                            'NumCatalogPurchases': catalog_purchases,
                            'NumWebPurchases': web_purchases,
                            'NumStorePurchases': store_purchases,
                            'Kidhome': 1 if has_children else 0,
                            'AcceptedCmp1': 1 if previous_campaign else 0,
                            'Total_Spending': total_spending
                        }
                        
                        # Convert to a list in the correct order
                        # This is a critical step - features must match the training data
                        st.warning("This is a simplified prediction interface. In production, you would need to ensure all features match exactly what was used in training.")
                        
                        # Create a vector with predicted features (simplified)
                        feature_vector = [0] * len(st.session_state.feature_names)
                        for i, feature_name in enumerate(st.session_state.feature_names):
                            if feature_name in features:
                                feature_vector[i] = features[feature_name]
                        
                        # Make prediction
                        prediction, probability = predict_response(
                            st.session_state.model, 
                            st.session_state.scaler, 
                            feature_vector
                        )
                        
                        # Display result with nice formatting
                        st.subheader("Prediction Result")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if prediction == 1:
                                st.success("Customer is likely to respond to the campaign")
                            else:
                                st.error("Customer is unlikely to respond to the campaign")
                        
                        with col2:
                            # Create a gauge chart for probability
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.barh(["Response Probability"], [probability], color='green')
                            ax.barh(["Response Probability"], [1-probability], left=[probability], color='red')
                            ax.set_xlim(0, 1)
                            for i, v in enumerate([probability]):
                                ax.text(v/2, i, f"{v:.1%}", color='white', va='center', ha='center')
                                ax.text(v + (1-v)/2, i, f"{1-v:.1%}", color='white', va='center', ha='center')
                            st.pyplot(fig)
                        
                        # Additional analysis
                        st.subheader("Marketing Recommendation")
                        if probability > 0.7:
                            st.success("High-value target! Consider premium offers or personalized outreach.")
                        elif probability > 0.4:
                            st.info("Potential responder. Include in regular campaign with standard incentives.")
                        else:
                            st.warning("Low response probability. May require special incentives or consider excluding from this campaign.")
else:
    st.info("Please upload your dataset to begin analysis")
    # Sample dashboard preview
    st.subheader("Dashboard Preview")
    st.image("https://via.placeholder.com/800x400?text=Marketing+Campaign+Analysis+Dashboard", use_column_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Marketing Campaign Analysis Tool")