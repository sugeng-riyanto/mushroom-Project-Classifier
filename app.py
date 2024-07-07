import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Title of the app
st.title("Mushroom Classification App")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the file
    df = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.write("Dataframe:", df.head())

    # Data cleansing
    st.write("Cleansing Data...")

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        st.write("Handling missing values...")
        df = df.dropna()
        st.write("Missing values handled. Data shape after dropping missing values:", df.shape)

    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

    st.write("Data after encoding categorical variables:")
    st.write(df.head())

    # Get the column names
    columns = df.columns.tolist()

    # Slider to select columns
    selected_columns = st.multiselect("Select columns to visualize", columns, default=columns)

    if selected_columns:
        # Plot the selected columns
        st.write("Visualizing Selected Columns")
        for col in selected_columns:
            st.write(f"Distribution of {col}")
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

    # Prepare data for modeling
    if st.checkbox("Train Model"):
        # Separate features and target
        X = df.drop('class', axis=1)
        y = df['class']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameter tuning using Grid Search with Cross-Validation
        param_grid = {
            'randomforestclassifier__n_estimators': [100, 200, 300],
            'randomforestclassifier__max_depth': [None, 10, 20, 30],
            'randomforestclassifier__min_samples_split': [2, 5, 10],
            'randomforestclassifier__min_samples_leaf': [1, 2, 4],
        }

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('randomforestclassifier', RandomForestClassifier(random_state=42))
        ])

        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=StratifiedKFold(5), n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Predict and evaluate
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

        # ROC Curve
        y_score = best_model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        roc_auc = auc(fpr, tpr)

        # Plotting ROC curve
        st.write("ROC Curve:")
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--')
        ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:0.2f})')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # Feature Importance
        st.write("Feature Importance:")
        feature_importances = best_model.named_steps['randomforestclassifier'].feature_importances_
        features = X.columns
        feature_importance_df = pd.DataFrame({'feature': features, 'importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), ax=ax)
        plt.title('Top 20 Feature Importances')
        st.pyplot(fig)

        # Save model and encoder
        st.session_state['model'] = best_model
        st.session_state['encoder'] = label_encoders

    # Prediction
    if 'model' in st.session_state and 'encoder' in st.session_state:
        st.write("Enter mushroom features to predict the class")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.text_input(f"Enter value for {col}")

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df)

            # Ensure input_df has same columns as X_train
            input_df = input_df.reindex(columns=X.columns, fill_value=0)

            for col in input_df.columns:
                if col in st.session_state['encoder']:
                    input_df[col] = st.session_state['encoder'][col].transform(input_df[col])

            prediction = st.session_state['model'].predict(input_df)
            predicted_class = label_encoders['class'].inverse_transform(prediction)
            st.write(f"The predicted class is: {predicted_class[0]}")
