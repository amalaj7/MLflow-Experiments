import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

mlflow.set_experiment("Pipeline Experiments")

# Loading the dataset
df = pd.read_csv('kaggle_diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df.copy(deep=True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)


mlflow.sklearn.autolog()
with mlflow.start_run():
    # Model Building
    X = df.drop(columns='Outcome')
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Creating Random Forest Model

    classifier = Pipeline([("scaler", StandardScaler()), ("lr", RandomForestClassifier(n_estimators=150,
                                                                                       n_jobs=-1,
                                                                                       random_state=30,
                                                                                       verbose=1))])
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print('Classification Report :\n', classification_report(y_test, y_pred))
    print('Test Accuracy Score : ', accuracy_score(y_test, y_pred))