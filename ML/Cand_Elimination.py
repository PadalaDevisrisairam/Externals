import pandas as pd

# Load CSV
data = pd.read_csv('path')

# Separate features (X) and labels (Y)
X = data.iloc[:, :-1].values.tolist()
Y = data.iloc[:, -1].values.tolist()

# Initialize Specific Hypothesis (S) and General Hypothesis (G)
S = X[0].copy()
G = [['?' for _ in range(len(S))]]

# Candidate Elimination Algorithm
for i, example in enumerate(X):
    if Y[i] == "Yes":
        for j in range(len(S)):
            if S[j] != example[j]:
                S[j] = '?'
        # Prune G
        G = [g for g in G if all(g[j] == '?' or g[j] == S[j] for j in range(len(S)))]

    elif Y[i] == "No":
        new_G = []
        for g in G:
            if not all(g[j] == '?' or g[j] == example[j] for j in range(len(S))):
                new_G.append(g)
            else:
                for j in range(len(S)):
                    if g[j] == '?' and S[j] != example[j]:
                        specialized = g[:]
                        specialized[j] = S[j]
                        if specialized not in new_G:
                            new_G.append(specialized)
        G = new_G

# Output
print("Final Specific Hypothesis (S):")
print(S)

print("\nFinal General Hypotheses (G):")
for g in G:
    print(g)
