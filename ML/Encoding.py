import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("/content/encoding_dataset.csv")
print("Original DataFrame:\n", df)

df_onehot = pd.get_dummies(df, columns=['Department', 'Gender'], dtype=int)

label_encoder = LabelEncoder()

df['Department_Label'] = label_encoder.fit_transform(df['Department'])
df['Gender_Label'] = label_encoder.fit_transform(df['Gender'])

print("\nAfter Label Encoding:\n", df)


print("\nAfter One-Hot Encoding:\n", df_onehot)
