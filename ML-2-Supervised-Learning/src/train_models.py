"""
train_models.py
----------------
Defines and trains the 5 required supervised learning algorithms, each
wrapped in its own sklearn Pipeline([preprocessor, classifier]) so that the
SAME fitted preprocessing logic is applied consistently and safely
(fit on train, transform on train/test) for every model.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def get_models() -> dict:
    """
    Return a dict of {model_name: sklearn estimator}. Estimators requiring
    a fixed seed use random_state=42. Reasonable, non-exotic hyperparameters
    are used throughout (no heavy tuning, per task instructions).
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "SVM": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
    }


def build_pipelines(preprocessor) -> dict:
    """
    Wrap each model in a Pipeline with the shared preprocessor, so that
    calling .fit(X_train, y_train) safely fits preprocessing + model
    together on the training data only, and .predict(X_test) reuses the
    already-fitted preprocessing.
    """
    pipelines = {}
    for name, model in get_models().items():
        pipelines[name] = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])
    return pipelines


def train_all(pipelines: dict, X_train, y_train) -> dict:
    """Fit every pipeline on the SAME training data. Returns fitted pipelines."""
    fitted = {}
    for name, pipeline in pipelines.items():
        print(f"Training {name} ...")
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
    return fitted
