
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ----------------------------------
# 1. Load dataset
# ----------------------------------

data = pd.read_csv("spam.csv", encoding="latin-1")

# Select required columns
data = data.iloc[:, :2]
data.columns = ["label", "message"]

data = data.dropna()

# ----------------------------------
# 2. Prepare labels
# ----------------------------------

data["label"] = data["label"].str.strip().str.lower()

data = data[data["label"].isin(["spam", "ham"])]

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

print("Dataset loaded successfully!")
print("Total messages:", len(data))

# ----------------------------------
# 3. Split dataset
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# ----------------------------------
# 4. TF-IDF feature extraction
# ----------------------------------

vectorizer = TfidfVectorizer(stop_words="english")

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# ----------------------------------
# 5. Create machine learning models
# ----------------------------------

models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "SVM": LinearSVC()
}

results = {}

best_model = None
best_model_name = None
best_f1 = -1

# ----------------------------------
# 6. Train and evaluate models
# ----------------------------------

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_vector, y_train)

    predictions = model.predict(X_test_vector)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    # Store performance results
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

    print("\nModel:", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Select model using F1-score
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# ----------------------------------
# 7. Performance comparison chart
# ----------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

naive_bayes_scores = [
    results["Naive Bayes"][metric]
    for metric in metrics
]

logistic_scores = [
    results["Logistic Regression"][metric]
    for metric in metrics
]

svm_scores = [
    results["SVM"][metric]
    for metric in metrics
]

x = range(len(metrics))
width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width for i in x],
    naive_bayes_scores,
    width,
    label="Naive Bayes"
)

plt.bar(
    x,
    logistic_scores,
    width,
    label="Logistic Regression"
)

plt.bar(
    [i + width for i in x],
    svm_scores,
    width,
    label="SVM"
)

plt.xticks(list(x), metrics)
plt.ylabel("Score")
plt.ylim(0, 1)

plt.title("Email Spam Classifier - Model Performance")

plt.legend()
plt.tight_layout()

plt.savefig("model_comparison.png")

plt.show()

# ----------------------------------
# 8. Save best model and vectorizer
# ----------------------------------

with open("model.pkl", "wb") as file:
    pickle.dump(best_model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\n--------------------------------")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("--------------------------------")

print("Best Model:", best_model_name)
print("Best F1-score:", best_f1)

print("\nModel saved as model.pkl")
print("Vectorizer saved as vectorizer.pkl")
print("Graph saved as model_comparison.png")
