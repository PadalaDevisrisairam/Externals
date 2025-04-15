from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# Function to load the dataset
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns = iris.feature_names)
    y = iris.target
    return X, y

def remove_duplicates(X, y):
    
    X['target'] = y
    print("Before Removing duplicates: ", X.shape)
    
    X = X.drop_duplicates()
    
    print("After Removing duplicates: ", X.shape)
    
    y = X['target']
    X = X.drop('target', axis=1)
    
    return X, y

def perform_cross_validation(X, y):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    model = KNeighborsClassifier(n_neighbors=5)
    
    # Perform cross-validation and get the accuracy scores for each fold
    scores = cross_val_score(model, X, y, cv=kfold)
    
    return scores

# Function to calculate the bias and variance from the cross-validation scores
def calculate_bias_variance(scores):
    mean_accuracy = np.mean(scores)
    variance = np.var(scores)
    
    print(f"Cross-validation scores: {scores}")
    print("Mean Accuracy (Low Bias): ", mean_accuracy)
    print("Variance of the cross-validation scores: ", variance)
    
    return mean_accuracy, variance

def main():
  X, y = load_data()
  X, y = remove_duplicates(X, y)
  scores = perform_cross_validation(X, y)
  calculate_bias_variance(scores)

main()
