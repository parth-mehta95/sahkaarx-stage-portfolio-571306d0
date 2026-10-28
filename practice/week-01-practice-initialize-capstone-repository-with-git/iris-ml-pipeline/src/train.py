"""
Model training and pipeline execution module for Iris ML Pipeline.

Supports training and evaluating both baseline LogisticRegression
and ensemble RandomForestClassifier models.
"""

import argparse
import os
from typing import Any, Tuple
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Support both relative and package-level imports
try:
    from .data_loader import load_iris, preprocess
    from .evaluate import evaluate_model, format_evaluation_report
except ImportError:
    from data_loader import load_iris, preprocess
    from evaluate import evaluate_model, format_evaluation_report


def train_model(
    model_type: str = "logistic_regression",
    test_size: float = 0.2,
    random_state: int = 42,
    model_output_path: str = "models/model.pkl",
) -> Tuple[Any, dict]:
    """Train and evaluate the specified Iris classification model.

    Parameters
    ----------
    model_type : str, default="logistic_regression"
        Choice of classifier: 'logistic_regression' or 'random_forest'.
    test_size : float, default=0.2
        Test split proportion.
    random_state : int, default=42
        Reproducibility seed.
    model_output_path : str, default="models/model.pkl"
        Path to serialize the trained model artifact.

    Returns
    -------
    Tuple[Any, dict]
        Fitted model instance and evaluation metrics dictionary.
    """
    # 1. Load data
    X, y = load_iris(as_frame=True, return_X_y=True)

    # 2. Preprocess and split without data leakage
    X_train, X_test, y_train, y_test = preprocess(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Initialize model based on configuration
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=200,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Choose 'logistic_regression' or 'random_forest'."
        )

    # 4. Fit model
    model.fit(X_train, y_train)

    # 5. Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    print(f"[{model_type.upper()}] Test Evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 6. Save model artifact
    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved to: {model_output_path}")

    return model, metrics


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Iris ML Pipeline Model Trainer")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "random_forest"],
        help="Model architecture to train (default: logistic_regression)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of test split (default: 0.2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/model.pkl",
        help="File path to save the serialized model artifact",
    )
    args = parser.parse_args()

    train_model(
        model_type=args.model,
        test_size=args.test_size,
        model_output_path=args.output,
    )


if __name__ == "__main__":
    main()
