from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    num_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])



def build_classification_pipeline(X):
    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", RandomForestClassifier(random_state=42))
    ])



def build_regression_pipeline(X):
    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", LinearRegression())
    ])