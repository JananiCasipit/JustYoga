import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_classifier(data_path='yoga_pose_data.csv', model_save_path='random_forest_pose_model.pkl'):
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {data_path}. Please collect data first.")
        return

    # Check if the label column exists
    if 'label' not in df.columns:
        print("❌ Error: 'label' column not found in the dataset.")
        return

    # Drop rows with missing or invalid data
    df = df.dropna()

    # Split features and target
    X = df.drop('label', axis=1)
    y = df['label']

    # Check class distribution
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        print(f"❌ Error: At least one pose label has fewer than 2 samples:\n{class_counts}")
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, model_save_path)
    print(f"✅ Model trained and saved to {model_save_path}")

def load_classifier(model_path='random_forest_pose_model.pkl'):
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}. Please train the model first.")
        return None

if __name__ == "__main__":
    train_classifier()
