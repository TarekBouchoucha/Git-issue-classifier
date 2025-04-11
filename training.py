import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)
import pickle

def load_data(processed_data_path):
    df = pd.read_csv(processed_data_path)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    return X, y

def train_random_forest(X_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=400, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    try:
        if len(np.unique(y_test)) > 2:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
        else:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print("\nROC-AUC Score:", roc_auc)
    except Exception as e:
        print("\nROC-AUC Score could not be computed:", str(e))

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def main():
    processed_data_path = 'processed_data_2.csv'
    model_path = 'random_forest_model_2.pkl'

    X, y = load_data(processed_data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = train_random_forest(X_train, y_train)

    evaluate_model(rf_model, X_test, y_test)

    save_model(rf_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
