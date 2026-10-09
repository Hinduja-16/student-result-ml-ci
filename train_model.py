
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 7, 9],
    "attendance": [40, 50, 55, 65, 70, 75, 80, 90, 45, 68, 85, 95],
    "result": [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]
})

X = data[["study_hours", "attendance"]]
y = data["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, "student_result_model.pkl")

metrics = {
    "accuracy": round(float(accuracy), 4),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

data.to_csv("student_results.csv", index=False)

print("Model and artifacts created successfully.")
print(metrics)
