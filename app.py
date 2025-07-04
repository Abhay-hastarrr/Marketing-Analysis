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
    page_title="Customer conversion prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .reportview-container {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        width: 100%;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        border-radius: 0.5rem;
    }
    .stSelectbox>div>div>select {
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #007bff;
    }
</style>
""", unsafe_allow_html=True)

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
        df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                 'MntFishProducts', 'MntSweetProducts']].sum(axis=1)
        # Remove outliers using Z-score
        numerical_cols = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                         'MntSweetProducts','NumDealsPurchases', 'NumWebPurchases', 
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
st.title("📊 Customer conversion prediction")

# Sidebar navigation with icons
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Campaign Analysis", 
    "Channel Effectiveness", 
    "Product Analysis", 
    "Customer Demographics", 
    "Predictive Model"
])

# Load data section
with st.expander("📂 Data Upload & Overview", expanded=True):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        data = load_data()
    
    if data is not None:
        # Clean data toggle
        clean_data_toggle = st.checkbox("Clean Data", value=True, key="clean_data_main")
        if clean_data_toggle:
            data = clean_data(data)
        
        # Display basic data info
        st.markdown("### Dataset Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Rows:** {data.shape[0]}")
        with col2:
            st.markdown(f"**Columns:** {data.shape[1]}")
        with col3:
            st.markdown(f"**Missing Values:** {data.isnull().sum().sum()}")
        
        # Show raw data if toggled
        if st.checkbox("Show Raw Data"):
            st.dataframe(data.head(10))
    else:
        st.info("Please upload your dataset to begin analysis")
        st.image("https://via.placeholder.com/800x400?text=Marketing+Campaign+Analysis+Dashboard", use_column_width=True)

# Page content based on navigation selection
if data is not None:
    # Data Overview Page
    if page == "Home":
        st.header("Data Overview")
        
        st.markdown("### Numeric Column Distribution")
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_col = st.selectbox("Select column to visualize", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data[selected_col], kde=True, ax=ax, color="#007bff")
        ax.set_title(f'Distribution of {selected_col}')
        st.pyplot(fig)
        
        st.markdown("### 🔍 Correlation Analysis")
        corr = data[numeric_cols].corr()

        # Adjust figure size based on number of columns
        fig_size = max(10, len(numeric_cols)), max(8, len(numeric_cols) * 0.7)
        fig, ax = plt.subplots(figsize=fig_size)

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Draw the heatmap with better font sizes and spacing
        sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .75},
        annot_kws={"size": 10},  # Reduce annotation font size
        ax=ax
        )

        # Improve label readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

        # Tight layout to avoid clipping
        plt.tight_layout()

        # Show plot
        st.pyplot(fig)

    # Campaign Analysis Page
    elif page == "Campaign Analysis":
        st.header("Campaign Analysis")

        st.markdown("### Campaign Engagement Rates")
        campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
        engagement_rates = data[campaign_columns].mean() * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=engagement_rates.index, y=engagement_rates.values, palette="Blues_d", ax=ax)
        ax.set_ylabel("Engagement Rate (%)")
        ax.set_title("Engagement Rates Across Campaigns")

        for i, v in enumerate(engagement_rates.values):
            ax.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom')

         # Fix: Add padding to the top of y-axis so text stays inside
        ax.set_ylim(top=max(engagement_rates.values) * 1.35)

        st.pyplot(fig)

        st.markdown("### Response by Age Group")
        bins = [0, 30, 40, 50, 60, 100]
        labels = ['<30', '30-40', '40-50', '50-60', '>60']
        data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)
        conversion_by_age = data.groupby('Age_Group')['Response'].mean() * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=conversion_by_age.index, y=conversion_by_age.values, palette="Blues_d", ax=ax)
        ax.set_ylabel("Conversion Rate (%)")
        ax.set_title("Conversion Rates by Age Group")

        for i, v in enumerate(conversion_by_age.values):
            ax.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom')

        # Fix: Add padding to the top of y-axis
        ax.set_ylim(top=max(conversion_by_age.values) * 1.15)

        st.pyplot(fig)

        st.markdown("### Response by Marital Status")
        conversion_by_status = data.groupby('Marital_Status')['Response'].mean() * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=conversion_by_status.index, y=conversion_by_status.values, palette="Blues_d", ax=ax)
        ax.set_ylabel("Conversion Rate (%)")
        ax.set_title("Conversion Rates by Marital Status")
        ax.tick_params(axis='x', rotation=45)

        for i, v in enumerate(conversion_by_status.values):
            ax.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom')

        # Fix: Add padding to the top of y-axis
        ax.set_ylim(top=max(conversion_by_status.values) * 1.15)

        st.pyplot(fig)

    # Channel Effectiveness Page
    elif page == "Channel Effectiveness":
        st.header("Channel Effectiveness")
        
        st.markdown("### Conversion Rates by Channel")
        avg_web_visits = data['NumWebVisitsMonth'].mean()
        avg_web_purchases = data['NumWebPurchases'].mean()
        avg_catalog_purchases = data['NumCatalogPurchases'].mean()
        avg_store_purchases = data['NumStorePurchases'].mean()
        
        conversion_rate_web = (avg_web_purchases / avg_web_visits) * 100
        conversion_rate_catalog = (avg_catalog_purchases / avg_web_visits) * 100
        conversion_rate_store = (avg_store_purchases / avg_web_visits) * 100
        
        conversion_rates = pd.DataFrame({
            'Channel': ['Web', 'Catalog', 'Store'],
            'Conversion Rate (%)': [conversion_rate_web, conversion_rate_catalog, conversion_rate_store]
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=conversion_rates, x='Channel', y='Conversion Rate (%)', palette="viridis", ax=ax)
        ax.set_title("Channel Conversion Rates")
        for i, v in enumerate(conversion_rates['Conversion Rate (%)']):
            ax.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom')
        st.pyplot(fig)
        
        st.markdown("### Channel Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">Avg Web Visits : {avg_web_visits:.1f}</div>
                    Avg Web Visits/Month
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">avg web purchases : {avg_web_purchases:.1f}</div>
                    Avg Web Purchases
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">conversion rate web : {conversion_rate_web:.1f}%</div>
                    Web Conversion Rate
                </div>
            """, unsafe_allow_html=True)
            
        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">avg catalog purchases : {avg_catalog_purchases:.1f}</div>
                    Avg Catalog Purchases
                </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">avg store purchases : {avg_store_purchases:.1f}</div>
                    Avg Store Purchases
                </div>
            """, unsafe_allow_html=True)
        with col6:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">conversion rate store : {conversion_rate_store:.1f}%</div>
                    Store Conversion Rate
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### Channel Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.pie(conversion_rates['Conversion Rate (%)'], labels=conversion_rates['Channel'], 
               autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis"))
        ax.set_title("Channel Conversion Distribution")
        st.pyplot(fig)

    # Product Analysis Page
    elif page == "Product Analysis":
        st.header("Product Analysis")
        
        st.markdown("### Total Sales by Product Category")
        total_sales = {
            'Wines': data['MntWines'].sum(),
            'Fruits': data['MntFruits'].sum(),
            'Meat Products': data['MntMeatProducts'].sum(),
            'Fish Products': data['MntFishProducts'].sum(),
            'Sweet Products': data['MntSweetProducts'].sum()
        }
        
        total_sales_df = pd.DataFrame(list(total_sales.items()), columns=['Product', 'Total_Sales'])
        total_sales_df = total_sales_df.sort_values(by='Total_Sales', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Total_Sales', y='Product', data=total_sales_df, palette="magma", ax=ax)
        ax.set_title("Total Sales by Product Category")
        for i, v in enumerate(total_sales_df['Total_Sales']):
            ax.text(v * 0.95, i, f"${v:,.0f}", ha='right', va='center', color='white', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("### Product Sales by Demographic")
        demographic_option = st.selectbox("Analyze Product Sales by:", ["Age Group", "Marital Status", "Education"])
        
        if demographic_option == "Age Group":
            if 'Age_Group' not in data.columns:
                bins = [0, 30, 40, 50, 60, 100]
                labels = ['<30', '30-40', '40-50', '50-60', '>60']
                data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)
            group_sales = data.groupby('Age_Group')[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                                'MntFishProducts', 'MntSweetProducts']].sum()
            st.markdown("#### Age Group Analysis")
        elif demographic_option == "Marital Status":
            group_sales = data.groupby('Marital_Status')[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                                    'MntFishProducts', 'MntSweetProducts']].sum()
            st.markdown("#### Marital Status Analysis")
        else:
            group_sales = data.groupby('Education')[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                              'MntFishProducts', 'MntSweetProducts']].sum()
            st.markdown("#### Education Level Analysis")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        group_sales.plot(kind='bar', stacked=True, ax=ax, colormap="tab20")
        plt.legend(title="Product Category")
        plt.title(f"Product Sales by {demographic_option}")
        st.pyplot(fig)

    # Customer Demographics Page
    elif page == "Customer Demographics":
        st.header("Customer Demographics")
        
        st.markdown("### Age & Income Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data['Age'], bins=30, kde=True, ax=ax, color="#007bff")
            ax.set_title("Age Distribution")
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data['Income'], bins=30, kde=True, ax=ax, color="#28a745")
            ax.set_title("Income Distribution")
            st.pyplot(fig)
            
        st.markdown("### Demographic Breakdown")
        col3, col4 = st.columns(2)
        
        with col3:
            education_counts = data['Education'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=education_counts.index, y=education_counts.values, ax=ax, palette="viridis")
            ax.set_title("Education Level Distribution")
            ax.tick_params(axis='x', rotation=45)
            for i, v in enumerate(education_counts.values):
                ax.text(i, v + 5, str(v), ha='center', va='bottom')
            st.pyplot(fig)
        
        with col4:
            marital_counts = data['Marital_Status'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=marital_counts.index, y=marital_counts.values, ax=ax, palette="plasma")
            ax.set_title("Marital Status Distribution")
            ax.tick_params(axis='x', rotation=45)
            for i, v in enumerate(marital_counts.values):
                ax.text(i, v + 5, str(v), ha='center', va='bottom')
            st.pyplot(fig)
            
        st.markdown("### Family Composition")
        col5, col6 = st.columns(2)
        
        with col5:
            kids_count = data['Kidhome'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(kids_count, labels=kids_count.index, autopct='%1.1f%%', startangle=90, 
                  colors=sns.color_palette("pastel"))
            ax.set_title("Customers with Children")
            st.pyplot(fig)
        
        with col6:
            teens_count = data['Teenhome'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(teens_count, labels=teens_count.index, autopct='%1.1f%%', startangle=90,
                  colors=sns.color_palette("Set2"))
            ax.set_title("Customers with Teenagers")
            st.pyplot(fig)
            
    # Predictive Model Page
    elif page == "Predictive Model":
        st.header("Predictive Model Training")
        
        st.markdown("""
        This section allows you to train a machine learning model to predict customer response to campaigns.
        The model will analyze patterns in your data to identify factors that influence campaign success.
        """)
        
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
                        
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.feature_names = feature_names
                        
                        if model is not None:
                            st.success("Model trained successfully!")
                            
                            st.markdown("### Model Performance")
                            st.text(classification_report(y_test, y_pred))
                                                
                            st.markdown("### Feature Importance")
                            fig, ax = plt.subplots(figsize=(12, 8))
                            sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15), palette="viridis", ax=ax)
                            plt.title("Top 15 Important Features")
                            st.pyplot(fig)
                            
                            col3, col4 = st.columns(2)
                            
                            with col3:
                                if st.button("Download Model"):
                                    joblib.dump(model, 'marketing_campaign_model.pkl')
                                    joblib.dump(scaler, 'feature_scaler.pkl')
                                    st.success("Model saved as 'marketing_campaign_model.pkl'")
                                    
                    except Exception as e:
                        st.error(f"An error occurred during model training: {e}")
        
        with predict_tab:
            st.markdown("### Prediction Interface")
            
            if st.session_state.model is None:
                st.warning("Please train a model first before making predictions")
            else:
                st.markdown("Enter customer information to predict campaign response likelihood:")
                
                with st.form("prediction_form"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        income = st.number_input("💰 Income", min_value=0, value=50000)
                        age = st.number_input("👤 Age", min_value=18, max_value=100, value=40)
                        total_spending = st.number_input("💳 Total Spending", min_value=0, value=500)
                        
                    with col2:
                        web_visits = st.number_input("🌐 Web Visits per Month", min_value=0, value=5)
                        catalog_purchases = st.number_input("📦 Catalog Purchases", min_value=0, value=2)
                        web_purchases = st.number_input("🛒 Web Purchases", min_value=0, value=3)
                        
                    with col3:
                        store_purchases = st.number_input("Store Purchases", min_value=0, value=3)
                        has_children = st.checkbox("Has Children")
                        previous_campaign = st.checkbox("Responded to Previous Campaign")
                    
                    submitted = st.form_submit_button("Predict Response")
                    
                    if submitted:
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
                        
                        feature_vector = [0] * len(st.session_state.feature_names)
                        for i, feature_name in enumerate(st.session_state.feature_names):
                            if feature_name in features:
                                feature_vector[i] = features[feature_name]
                        
                        prediction, probability = predict_response(
                            st.session_state.model, 
                            st.session_state.scaler, 
                            feature_vector
                        )
                        
                        st.markdown("### Prediction Result")
                        col_result, col_gauge = st.columns([2, 1])
                        
                        with col_result:
                            if prediction == 1:
                                st.markdown("<div class='metric-card' style='background-color: #d4edda; color: #155724;'>" +
                                           "✅ visitor is likely to buy something" +
                                           "</div>", unsafe_allow_html=True)
                            else:
                                st.markdown("<div class='metric-card' style='background-color: #f8d7da; color: #721c24;'>" +
                                           "visitor is unlikely to buy something" +
                                           "</div>", unsafe_allow_html=True)
                        
                        with col_gauge:
                            fig, ax = plt.subplots(figsize=(4, 2))
                            ax.barh(["Response Probability"], [probability], color='#28a745')
                            ax.barh(["Response Probability"], [1-probability], left=[probability], color='#dc3545')
                            ax.set_xlim(0, 1)
                            ax.axis('off')
                            for i, v in enumerate([probability]):
                                ax.text(v/2, i, f"{v:.1%}", color='white', va='center', ha='center')
                                ax.text(v + (1-v)/2, i, f"{1-v:.1%}", color='white', va='center', ha='center')
                            st.pyplot(fig)
                        
                        st.markdown("### 📢 Marketing Recommendation")
                        if probability > 0.7:
                            st.markdown("<div class='metric-card' style='background-color: #cce5ff; color: #004085;'>" +
                                       "🌟 High-value target! Consider premium offers or personalized outreach." +
                                       "</div>", unsafe_allow_html=True)
                        elif probability > 0.4:
                            st.markdown("<div class='metric-card' style='background-color: #fff3cd; color: #856404;'>" +
                                       "💡 Potential responder. Include in regular campaign with standard incentives." +
                                       "</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='metric-card' style='background-color: #f8d7da; color: #721c24;'>" +
                                       "⚠️ Low response probability. May require special incentives or consider excluding from this campaign." +
                                       "</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("---")
    