Salary Prediction using Machine Learning

Project Overview

This project builds a machine learning model to predict an individual's
salary using demographic, professional, and educational information.

The project is implemented in Python using a Jupyter Notebook and
follows a practical machine learning workflow:

Data loading and inspection

Exploratory Data Analysis (EDA)

Data cleaning

Duplicate and missing-value handling

Categorical feature encoding

Feature/target separation

Train-test splitting

Feature scaling

Linear Regression model training

Model evaluation

Saving the trained model and scaler

The main notebook is Salary_prediction.ipynb.

Dataset

The project uses a CSV dataset named Salary.csv.

The original dataset contains 2,000 records and 4 columns:

Feature             Description                        Type

Age               Age of the individual              Numerical
YearsExperience   Years of professional experience   Numerical
EducationLevel    Highest education level            Categorical
Salary            Salary to be predicted             Target

During preprocessing, the columns were renamed to:

Age

Experience_year

Education_level

Salary

After duplicate removal, the working dataset contains 1,999 records.
The notebook reports no missing values after cleaning.

Exploratory Data Analysis

The notebook performs basic EDA to understand the structure and
relationships within the data.

Key observations from the analysis:

Average age: approximately 41.08 years

Average experience: approximately 12.94 years

Average salary: approximately 102,085.98

Salary range: 29,100 to 221,269

Experience has a strong positive correlation with salary in this
dataset, with a reported correlation of approximately 0.96.

Age has a reported correlation of approximately 0.64 with
salary.

The education-level distribution in the cleaned data is:

Education Level     Count

Bachelor              909
Master                481
HighSchool            401
PhD                   208

These relationships are descriptive of the dataset and should not be
interpreted as causal effects.

Data Preprocessing

1. Duplicate Removal

Duplicate records were identified and removed using:

Sal_data = Sal_data.drop_duplicates(keep="first")

One duplicate record was identified, reducing the dataset from 2,000 to
1,999 rows.

2. Missing-Value Handling

Missing values were checked and handled using:

Sal_data.isnull().sum()
Sal_data.dropna(how="any", inplace=True)

The notebook reports zero missing values.

3. Categorical Encoding

Education_level is a categorical feature, so one-hot encoding was
applied:

Sal_data = pd.get_dummies(
    Sal_data,
    columns=["Education_level"],
    drop_first=True
)

This produces the following encoded education features:

Education_level_HighSchool

Education_level_Master

Education_level_PhD

With drop_first=True, the remaining education category acts as the
reference category.

4. Feature and Target Selection

The target variable is:

Salary

The predictor variables are:

Age
Experience_year
Education_level_HighSchool
Education_level_Master
Education_level_PhD

The notebook separates them using:

x = Sal_data.drop("Salary", axis=1)
y = Sal_data["Salary"]

5. Train-Test Split

The dataset is divided into training and testing sets using an 80/20
split:

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.20,
    random_state=42
)

This provides a reproducible train-test split.

Machine Learning Model

The project uses Linear Regression from Scikit-learn:

from sklearn.linear_model import LinearRegression

Linear_regression_model = LinearRegression()
Linear_regression_model.fit(x_train, y_train)

Linear Regression is appropriate as a baseline regression approach
because the target variable, Salary, is continuous.

Model Evaluation

The notebook evaluates the trained model using:

R² Score

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Reported evaluation results:

Metric         Result

R² Score       0.9697
MAE          5,503.40
RMSE         6,973.70

These results are based on the notebook's recorded test-set evaluation.
Performance can vary if the data, preprocessing, split, or environment
changes.

Model Persistence

The trained model and scaler are saved using Joblib:

import joblib

joblib.dump(Linear_regression_model, "salary_model.pkl")
joblib.dump(scaler, "salary_scaler.pkl")

Generated artifacts:

salary_model.pkl
salary_scaler.pkl

These files can be loaded later for inference without retraining the
model.


Key Takeaways

The dataset contains salary information associated with age,
experience, and education level.

Experience shows a strong relationship with salary in the analyzed
dataset.

Categorical education information was converted into numerical
features using one-hot encoding.

Linear Regression was used as the prediction model.

The recorded test evaluation achieved an R² score of approximately
0.97.

The model and scaler were persisted using Joblib for later use.

Limitations and Future Improvements

This project is a learning-oriented salary prediction model. For a
production-grade system, the following improvements would be valuable:

Compare Linear Regression with models such as Random Forest,
Gradient Boosting, and other regression algorithms.

Use cross-validation instead of relying on a single train-test
split.

Perform systematic hyperparameter tuning where applicable.

Add residual and error analysis.

Check for outliers and influential observations.

Build a reproducible preprocessing pipeline so training and
inference use exactly the same transformations.

Add automated tests for preprocessing and prediction.

Add a user-facing application using Streamlit or Flask.

Add model versioning and experiment tracking for production
workflows.

Monitor model performance when new data becomes available.

Technologies Used

Python

NumPy

Pandas

Seaborn

Matplotlib

Scikit-learn

Joblib

Jupyter Notebook

Author

Lipsa Mahakul

If you use or extend this project, feel free to build upon the notebook
and add a deployment layer for real-time salary prediction.

Disclaimer

This project is intended for educational and demonstration purposes.
Predictions from a machine learning model should not be treated as
guaranteed salary outcomes. Real-world compensation depends on many
factors that may not be represented in the dataset.
