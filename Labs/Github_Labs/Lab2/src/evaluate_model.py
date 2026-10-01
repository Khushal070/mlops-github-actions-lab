import argparse, json, os, sys
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

F1_THRESHOLD = 0.99

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    ts = args.timestamp

    model = joblib.load(f'model_{ts}_dt_model.joblib')

    X, y = load_breast_cancer(return_X_y=True)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "F1_Score": f1_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, proba),
    }

    with open(f'{ts}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # append to history.json
    metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    history_path = os.path.join(metrics_dir, 'history.json')
    history = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    history.append({"timestamp": ts, "model": "GradientBoostingClassifier", **metrics})
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    # quality gate
    if metrics["F1_Score"] < F1_THRESHOLD:
        print(f"F1 {metrics['F1_Score']:.3f} below {F1_THRESHOLD}, failing run")
        sys.exit(1)
    print(f"Quality gate passed: F1 {metrics['F1_Score']:.3f}")