import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("/content/house_prices_linear_regression.csv")

cols = df.columns.to_list()

encoder = LabelEncoder()
df['Location'] = encoder.fit_transform(df['Location'])

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

print("\n--- Sample Output Verification ---")
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted' : [round(val) for val in y_pred]
})
print(results)
