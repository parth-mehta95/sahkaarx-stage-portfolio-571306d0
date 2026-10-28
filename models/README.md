# Trained Model Artifacts Directory 🌸🤖

This directory houses the serialized model estimators, metadata manifests, and serialization utilities for the **Iris ML Pipeline Capstone**.

---

## 📌 Artifact Catalog

| Model Identifier | Architecture | Paradigm | Accuracy | Macro F1 | Artifact Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`decision_tree`** | `DecisionTreeClassifier(max_depth=3)` | Supervised (White-box) | **96.67%** | **0.9666** | [`models/decision_tree.joblib`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/models/decision_tree.joblib) |
| **`random_forest`** | `RandomForestClassifier(n_estimators=100, max_depth=3)` | Supervised (Ensemble) | **100.00%** | **1.0000** | [`models/random_forest.joblib`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/models/random_forest.joblib) |
| **`kmeans`** | `KMeans(n_clusters=3)` + Hungarian Alignment | Unsupervised (Centroid) | **86.67%** | **0.8658** | [`models/kmeans.joblib`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/models/kmeans.joblib) |

---

## 🛠️ Loading and Running Inference

All models are serialized using `joblib` and accept scaled 4-dimensional input feature vectors:

```python
import joblib
import numpy as np

# 1. Load trained estimator
model = joblib.load("models/decision_tree.joblib")

# 2. Prepare sample feature vector: [sepal_length, sepal_width, petal_length, petal_width]
# Note: Input features must be scaled using the fitted StandardScaler from preprocessing
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])

# 3. Predict class label
predicted_class = model.predict(sample_input)
class_names = ["setosa", "versicolor", "virginica"]
print(f"Predicted species: {class_names[predicted_class[0]]}")
```

---

## 🔄 Model Retraining & Export

To retrain and re-export the model binaries at any time:

```bash
python models/export_models.py
```
or run the root models module:
```bash
python models.py
```
