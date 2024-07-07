# Mushroom Classification App

This Streamlit app allows users to upload a dataset of mushrooms, perform data cleansing, train a RandomForestClassifier model, and make predictions about mushroom classes based on user inputs. The app includes data visualizations, model evaluation metrics, and feature importance analysis.
Visit this web to try:
    https://mushroomproject.streamlit.app/

## Features

- Upload a CSV file containing mushroom data.
- Cleanse the dataset by handling missing values and encoding categorical variables.
- Visualize the distribution of selected columns.
- Train a RandomForestClassifier model with hyperparameter tuning using GridSearchCV and cross-validation.
- Evaluate the model using accuracy score, confusion matrix, ROC curve, and feature importance.
- Make predictions based on user inputs.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/sugeng-riyanto/mushroom-Project-Classifier.git
    cd mushroom-Project-Classifier
    
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Upload your mushroom dataset in CSV format.
3. Select columns to visualize and analyze the distributions.
4. Train the model by checking the "Train Model" checkbox.
5. View model evaluation metrics including accuracy score, confusion matrix, ROC curve, and feature importance.
6. Make predictions by entering mushroom features and clicking the "Predict" button.

## Files

- `mushroom_classification_app.py`: The main Streamlit app script.
- `requirements.txt`: The required packages and their versions.

## Requirements

- Python 3.8 or higher
- Streamlit 1.24.0
- Pandas 1.3.5
- Matplotlib 3.5.1
- Seaborn 0.11.2
- Scikit-learn 1.0.2

## Example Dataset

The app expects a CSV file with the following columns:

- class
- cap-shape
- cap-surface
- cap-color
- bruises
- odor
- gill-attachment
- gill-spacing
- gill-size
- gill-color
- stalk-shape
- stalk-root
- stalk-surface-above-ring
- stalk-surface-below-ring
- stalk-color-above-ring
- stalk-color-below-ring
- veil-type
- veil-color
- ring-number
- ring-type
- spore-print-color
- population
- habitat

## License

This project is licensed under the ... License- see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)

