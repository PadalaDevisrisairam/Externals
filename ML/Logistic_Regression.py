import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/weather_forecast.csv")  

encoder = LabelEncoder()
cols = df.columns.to_list()

for col in cols:
    df[col] = encoder.fit_transform(df[col])

X = df.drop('play', axis=1)
y = df['play']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)

print("\n--- Sample Output Verification ---")
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
print(results)
