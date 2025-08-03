# iris_classifier.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Make our charts super pretty!
sns.set(style='darkgrid', font_scale=1.2, rc={"axes.facecolor": "#222b35", "figure.facecolor": "#14181e"})

# 1. Meet our flower friends (load the data)
iris = load_iris(as_frame=True)
df = iris['frame']
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("Here are the first few flowers I met:\n", df.head())

# 2. Let's draw! Paint the relationships.
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue='species', palette="winter", diag_kind="kde")
plt.suptitle("Iris Pairwise Feature Plots", color="#00ff99", fontsize=18, y=1.03)
plt.show()

colors = ["#63e5ef", "#8afcbf", "#febbbb"]
df.iloc[:, :-1].hist(figsize=(11, 7), color=colors, edgecolor='black')
plt.suptitle("Iris Feature Histograms", color="#ff82f7", fontsize=17)
plt.show()

# 3. Split up the flowers—some for learning, some for testing!
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# 4. Let the computer try different ways to guess
models = [
    ("KNN (asks neighbors)", KNeighborsClassifier(n_neighbors=5)),
    ("Logistic Regression (math brain)", LogisticRegression(max_iter=200)),
    ("Decision Tree (asks questions)", DecisionTreeClassifier(random_state=42))
]
results = {}

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'conf_matrix': confusion_matrix(y_test, y_pred),
        'clf': model
    }
    print(f"\n{name} got {results[name]['accuracy']*100:.2f}% of the guesses right!")
    print(classification_report(y_test, y_pred))

# 5. Cheering: Show how well each computer method guessed
fig, axs = plt.subplots(1, 3, figsize=(18, 4))
for idx, (name, data) in enumerate(results.items()):
    sns.heatmap(data['conf_matrix'], annot=True, fmt='d', ax=axs[idx],
                cmap='cool', 
                xticklabels=iris.target_names, yticklabels=iris.target_names,
                cbar=False)
    axs[idx].set_title(f"{name} — Scoreboard")
    axs[idx].set_xlabel('I guessed')
    axs[idx].set_ylabel('Real answer')
plt.suptitle('How good was our computer at Flower Guessing?', color="#00fad8", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# 6. Announce the winner!
winner = max(results, key=lambda m: results[m]['accuracy'])
print(f"\n🏅 Out of all ways, '{winner}' was the best guesser with an accuracy of {results[winner]['accuracy']*100:.2f}%! 😃🌸")
