import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
accuracies = []

for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"--- Fold {fold} ---")
    for i in range(len(y_test)):
        actual = target_names[y_test[i]]
        predicted = target_names[y_pred[i]]
        if y_test[i] == y_pred[i]:
            print(f"✅ Correct: predicted = {predicted}, Actual = {actual}")
        else:
            print(f"❌ Wrong: predicted = {predicted}, Actual = {actual}")
    print("\n")

    fold += 1

print(f"Final Accuracy: {(sum(accuracies) / len(accuracies)*100):.2f}%")
