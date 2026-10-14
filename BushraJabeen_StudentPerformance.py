"""
Student Performance Analysis and Prediction Using AI
Prepared for: Bushra Jabeen
AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026

Note: The included data is synthetic demonstration data, not real student data.
Run: python BushraJabeen_StudentPerformance.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

DATA_PATH = Path(__file__).with_name("student_performance_dataset.csv")
OUTPUT_DIR = Path(__file__).with_name("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    data = pd.read_csv(DATA_PATH)
    print("Dataset shape:", data.shape)
    print("\nFirst five rows:")
    print(data.head())
    print("\nMissing values:")
    print(data.isnull().sum())
    print("\nSummary statistics:")
    print(data.describe(numeric_only=True))

    data["performance_category"].value_counts().plot(kind="bar", title="Student Performance Categories")
    plt.xlabel("Category")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "performance_categories.png", dpi=150)
    plt.close()

    data.plot.scatter(x="study_hours_per_day", y="final_score",
                      title="Study Hours vs Final Score", alpha=0.65)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "study_hours_vs_score.png", dpi=150)
    plt.close()

    features = [
        "study_hours_per_day", "attendance_percent", "previous_score",
        "assignments_completed_out_of_10", "sleep_hours_per_day"
    ]
    X = data[features]
    y = data["performance_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\nRandom Forest test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=["High", "Medium", "Needs Support"], xticks_rotation=20
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    importance = pd.Series(model.feature_importances_, index=features).sort_values()
    importance.plot(kind="barh", title="Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()

    example = pd.DataFrame([{
        "study_hours_per_day": 3.0,
        "attendance_percent": 85.0,
        "previous_score": 70.0,
        "assignments_completed_out_of_10": 7,
        "sleep_hours_per_day": 7.0
    }])
    print("\nExample prediction (hypothetical inputs):", model.predict(example)[0])
    print("\nCharts saved in:", OUTPUT_DIR)
    print("Reminder: synthetic data only; do not use this model to make decisions about real students.")

if __name__ == "__main__":
    main()
