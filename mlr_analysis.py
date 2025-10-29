from __future__ import annotations

import argparse
import os
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

MPLCONFIG_PATH = Path(".mplconfig")
MPLCONFIG_PATH.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_PATH.resolve()))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("AmesHousing .csv")
REPORT_PATH = Path("analysis_report.md")
PLOT_PATH = Path("prediction_plot.png")

DEFAULT_PARAMS = {
    "k_features": 40,
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "alpha": 0.05,
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def describe_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    missing_ratio = df.isna().mean().sort_values(ascending=False)
    high_missing = missing_ratio[missing_ratio > 0.4]
    numeric_summary = df.describe(include=[np.number])
    categorical_levels = {
        col: df[col].nunique(dropna=True)
        for col in df.select_dtypes(include=["object"]).columns
    }

    correlation = (
        df.select_dtypes(include=[np.number])
        .drop(columns=["PID", "Order"], errors="ignore")
        .corr(numeric_only=True)["SalePrice"]
        .dropna()
        .sort_values(ascending=False)
    )

    profile = {
        "shape": df.shape,
        "missing_ratio": missing_ratio,
        "high_missing": high_missing,
        "numeric_summary": numeric_summary,
        "categorical_levels": categorical_levels,
        "sale_price_summary": df["SalePrice"].describe(),
        "top_correlations": correlation.head(10),
        "bottom_correlations": correlation.tail(10),
    }
    return profile


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    profile = describe_dataset(df)
    drop_cols = list(profile["high_missing"].index)

    df_clean = df.drop(columns=drop_cols)
    y = df_clean["SalePrice"]
    X = df_clean.drop(columns=["SalePrice", "Order", "PID"], errors="ignore")

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preparation_details = {
        "dropped_columns": drop_cols,
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
    }

    return X, y, preparation_details


def make_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    k_features: int | str = "all",
) -> Pipeline:
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    selector = SelectKBest(score_func=f_regression, k=k_features)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("regressor", LinearRegression()),
        ]
    )

    return model


def get_feature_names(model: Pipeline) -> List[str]:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    feature_names: List[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            encoder: OneHotEncoder = (
                preprocessor.named_transformers_["cat"].named_steps["onehot"]
            )
            encoded_names = encoder.get_feature_names_out(columns)
            feature_names.extend(encoded_names.tolist())
    return feature_names


def normalize_k(k_requested: int | None, total_features: int) -> int | str:
    if k_requested is None or k_requested >= total_features:
        return "all"
    k_requested = max(1, k_requested)
    return k_requested


def compute_prediction_intervals(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    design_train = model[:-1].transform(X_train)
    design_eval = model[:-1].transform(X_eval)

    if hasattr(design_train, "toarray"):
        design_train = design_train.toarray()
    if hasattr(design_eval, "toarray"):
        design_eval = design_eval.toarray()

    n_samples, n_features = design_train.shape
    X_train_aug = np.hstack([np.ones((n_samples, 1)), design_train])
    XtX = X_train_aug.T @ X_train_aug
    ridge = np.eye(XtX.shape[0]) * 1e-6
    XtX_inv = np.linalg.inv(XtX + ridge)

    residuals = y_train - model.predict(X_train)
    df_resid = max(n_samples - n_features - 1, 1)
    sigma2 = float((residuals @ residuals) / df_resid)
    sigma2 = max(sigma2, 1e-12)
    t_value = float(stats.t.ppf(1 - alpha / 2, df_resid))

    X_eval_aug = np.hstack([np.ones((design_eval.shape[0], 1)), design_eval])
    predictions = model.predict(X_eval)

    se_mean = np.array(
        [np.sqrt(sigma2 * row @ XtX_inv @ row) for row in X_eval_aug],
        dtype=float,
    )
    se_pred = np.sqrt(se_mean**2 + sigma2)

    conf_lower = predictions - t_value * se_mean
    conf_upper = predictions + t_value * se_mean
    pred_lower = predictions - t_value * se_pred
    pred_upper = predictions + t_value * se_pred

    interval_df = pd.DataFrame(
        {
            "y_pred": predictions,
            "conf_lower": conf_lower,
            "conf_upper": conf_upper,
            "pred_lower": pred_lower,
            "pred_upper": pred_upper,
        },
        index=X_eval.index,
    )
    return interval_df


def get_selected_features_frame(model: Pipeline) -> pd.DataFrame:
    selector: SelectKBest = model.named_steps["selector"]
    regressor: LinearRegression = model.named_steps["regressor"]

    feature_names = get_feature_names(model)
    support = selector.get_support()
    scores = selector.scores_
    p_values = selector.pvalues_
    coefficients = regressor.coef_

    selected_names = [name for name, keep in zip(feature_names, support) if keep]
    selected_scores = scores[support] if scores is not None else np.full(len(selected_names), np.nan)
    selected_pvalues = (
        p_values[support] if p_values is not None else np.full(len(selected_names), np.nan)
    )

    df_features = pd.DataFrame(
        {
            "feature": selected_names,
            "score": selected_scores,
            "p_value": selected_pvalues,
            "coefficient": coefficients,
        }
    ).sort_values(by="score", ascending=False)

    return df_features


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    k_requested: int | None,
    test_size: float,
    random_state: int,
    alpha: float,
) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_pipeline(numeric_features, categorical_features, k_features="all")
    pipeline.fit(X_train, y_train)
    total_features = pipeline.named_steps["selector"].scores_.shape[0]
    k_effective = normalize_k(k_requested, total_features)

    if k_effective != "all":
        pipeline.set_params(selector__k=k_effective)
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    residuals = y_test - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    interval_df = compute_prediction_intervals(
        pipeline, X_train, y_train, X_test, alpha=alpha
    )

    prediction_df = pd.DataFrame({"SalePrice": y_test})
    prediction_df = prediction_df.join(interval_df)
    prediction_df["residual"] = residuals

    residual_stats = {
        "mean": float(residuals.mean()),
        "std": float(residuals.std()),
        "skew": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
        "q05": float(np.quantile(residuals, 0.05)),
        "q95": float(np.quantile(residuals, 0.95)),
    }

    selected_features = get_selected_features_frame(pipeline)

    results = {
        "model": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "prediction_df": prediction_df.sort_index(),
        "metrics": {
            "r2_test": r2,
            "rmse_test": rmse,
            "mae_test": mae,
        },
        "residual_stats": residual_stats,
        "selected_features": selected_features,
        "total_features": total_features,
        "effective_k": total_features if k_effective == "all" else k_effective,
    }
    return results


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    k_effective: int,
    cv_folds: int,
) -> Dict[str, Any]:
    selector_param: int | str = k_effective if k_effective < np.inf else "all"
    model = build_pipeline(
        numeric_features,
        categorical_features,
        k_features=selector_param if isinstance(selector_param, int) else "all",
    )

    r2_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="r2")
    mse_scores = -cross_val_score(model, X, y, cv=cv_folds, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(mse_scores)

    return {
        "r2_mean": float(r2_scores.mean()),
        "r2_std": float(r2_scores.std()),
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
        "fold_r2": r2_scores.tolist(),
        "fold_rmse": rmse_scores.tolist(),
    }


def extract_coefficients(model: Pipeline) -> pd.Series:
    feature_names = get_feature_names(model)
    selector: SelectKBest = model.named_steps["selector"]
    support = selector.get_support()
    selected_names = [name for name, keep in zip(feature_names, support) if keep]
    regressor: LinearRegression = model.named_steps["regressor"]
    coefficients = pd.Series(regressor.coef_, index=selected_names).sort_values(
        ascending=False
    )
    return coefficients


def create_prediction_plot(prediction_df: pd.DataFrame, alpha: float) -> plt.Figure:
    sorted_df = prediction_df.sort_values("SalePrice").reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    idx = np.arange(len(sorted_df))

    ax.plot(idx, sorted_df["SalePrice"], label="Actual", color="black", linewidth=1.2)
    ax.plot(idx, sorted_df["y_pred"], label="Predicted", color="tab:blue", linewidth=1.2)
    ax.fill_between(
        idx,
        sorted_df["pred_lower"],
        sorted_df["pred_upper"],
        color="tab:blue",
        alpha=0.2,
        label=f"{int((1 - alpha) * 100)}% prediction interval",
    )
    ax.fill_between(
        idx,
        sorted_df["conf_lower"],
        sorted_df["conf_upper"],
        color="tab:orange",
        alpha=0.2,
        label=f"{int((1 - alpha) * 100)}% confidence interval",
    )

    ax.set_xlabel("Sorted test observations")
    ax.set_ylabel("SalePrice")
    ax.set_title("Predictions with Confidence and Prediction Intervals")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    return fig


def run_analysis(
    k_features: int | None,
    test_size: float,
    random_state: int,
    cv_folds: int,
    alpha: float,
) -> Dict[str, Any]:
    df = load_data()
    dataset_profile = describe_dataset(df)
    X, y, preparation_details = prepare_features(df)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    training_results = train_model(
        X,
        y,
        numeric_features,
        categorical_features,
        k_requested=k_features,
        test_size=test_size,
        random_state=random_state,
        alpha=alpha,
    )

    cv_results = cross_validate_model(
        X,
        y,
        numeric_features,
        categorical_features,
        k_effective=training_results["effective_k"],
        cv_folds=cv_folds,
    )

    final_model = build_pipeline(
        numeric_features,
        categorical_features,
        k_features=training_results["effective_k"]
        if training_results["effective_k"] < training_results["total_features"]
        else "all",
    )
    final_model.fit(X, y)

    coefficients = extract_coefficients(final_model)
    selected_features_full = get_selected_features_frame(final_model)

    summary = {
        "dataset_profile": dataset_profile,
        "preparation": preparation_details,
        "train": training_results,
        "cv": cv_results,
        "coefficients": coefficients,
        "selected_features_full": selected_features_full,
        "parameters": {
            "k_features": training_results["effective_k"],
            "test_size": test_size,
            "random_state": random_state,
            "cv_folds": cv_folds,
            "alpha": alpha,
        },
    }
    return summary


def generate_report(summary: Dict[str, Any]) -> str:
    dataset_profile = summary["dataset_profile"]
    preparation = summary["preparation"]
    metrics = summary["train"]["metrics"]
    residual_stats = summary["train"]["residual_stats"]
    cv_results = summary["cv"]
    parameters = summary["parameters"]
    selected_features = summary["train"]["selected_features"].head(15)
    top_positive = summary["coefficients"].head(10)
    top_negative = summary["coefficients"].tail(10)

    lines: List[str] = [
        "# Ames Housing Linear Regression Analysis (CRISP-DM)",
        "",
        "## Business Understanding",
        "- **Goal**: Predict Ames, Iowa housing sale prices using structural and neighborhood attributes.",
        "- **Success Criteria**: Deliver a linear regression model with strong out-of-sample performance and interpretable feature impacts for pricing guidance.",
        "",
        "## Data Understanding",
        f"- **Rows × Columns**: {dataset_profile['shape'][0]} × {dataset_profile['shape'][1]}.",
        f"- **Target (`SalePrice`)**: mean ${dataset_profile['sale_price_summary']['mean']:,.0f}, median ${dataset_profile['sale_price_summary']['50%']:,.0f}, std ${dataset_profile['sale_price_summary']['std']:,.0f}.",
        "- **Top correlated features**: "
        + ", ".join([f"{idx} ({val:.2f})" for idx, val in dataset_profile["top_correlations"].items()]),
        "- **High-missing columns removed (>40% missing)**: "
        + (", ".join(preparation["dropped_columns"]) or "None"),
        "",
        "## Data Preparation",
        f"- Numeric features: {preparation['numeric_feature_count']}, categorical features: {preparation['categorical_feature_count']}.",
        "- Imputation: median for numeric, most-frequent for categorical.",
        "- Scaling: standardisation applied to numeric features.",
        "- Encoding: one-hot encoding with first level dropped per categorical feature.",
        f"- Feature selection: SelectKBest (`f_regression`) retaining top {parameters['k_features']} predictors.",
        "",
        "## Modeling & Evaluation",
        f"- Train/test split: {int((1 - parameters['test_size']) * 100)}% train / {int(parameters['test_size'] * 100)}% test (random_state={parameters['random_state']}).",
        "- Model: Ordinary Least Squares linear regression.",
        f"- Test R²: {metrics['r2_test']:.3f}",
        f"- Test RMSE: ${metrics['rmse_test']:,.0f}",
        f"- Test MAE: ${metrics['mae_test']:,.0f}",
        f"- CV ({parameters['cv_folds']} folds) R² mean ± std: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}",
        f"- CV RMSE mean ± std: ${cv_results['rmse_mean']:,.0f} ± ${cv_results['rmse_std']:,.0f}",
        f"- Residual diagnostics (test set): mean {residual_stats['mean']:.2f}, std {residual_stats['std']:.2f}, skew {residual_stats['skew']:.2f}, kurtosis {residual_stats['kurtosis']:.2f}, 5th pct {residual_stats['q05']:.2f}, 95th pct {residual_stats['q95']:.2f}.",
        "",
        "## Feature Insights",
        "- **Top selected features (by F-score)**:",
    ]

    for row in selected_features.itertuples():
        lines.append(
            f"  - {row.feature}: score {row.score:.2f}, p-value {row.p_value:.3e}, coefficient {row.coefficient:,.2f}"
        )

    lines.append(
        "- **Largest positive coefficients**: "
        + ", ".join([f"{idx} (+{val:,.1f})" for idx, val in top_positive.items()])
    )
    lines.append(
        "- **Largest negative coefficients**: "
        + ", ".join([f"{idx} ({val:,.1f})" for idx, val in top_negative.items()])
    )
    lines.extend(
        [
            "",
            "## Prediction Intervals",
            f"- Generated {int((1 - parameters['alpha']) * 100)}% confidence and prediction intervals for hold-out predictions to quantify uncertainty bands.",
            "",
            "## Recommendations",
            "- Use predicted price with interval bounds to communicate pricing risk and upside for stakeholders.",
            "- Investigate influential outliers (noted heavy tails in residuals) to refine fit or segment the market.",
            "- Extend with interaction terms or non-linear models to capture remaining variance while benchmarking against this interpretable baseline.",
        ]
    )

    return "\n".join(lines)


def save_report_and_plot(summary: Dict[str, Any], report_text: str) -> None:
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    fig = create_prediction_plot(summary["train"]["prediction_df"], summary["parameters"]["alpha"])
    fig.savefig(PLOT_PATH, dpi=144, bbox_inches="tight")
    plt.close(fig)


def run_streamlit_app(default_params: Dict[str, Any]) -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Streamlit is required for the interactive app. Install streamlit or run with --no-ui."
        ) from exc

    st.set_page_config(
        page_title="Ames Housing Linear Regression",
        layout="wide",
    )
    st.title("Ames Housing Linear Regression Analysis")
    st.write(
        "This app performs feature-selected multiple linear regression on the Ames Housing dataset "
        "and provides prediction intervals for estimated sale prices."
    )

    df = load_data()
    X, _, prep_details = prepare_features(df)
    total_feature_candidates = prep_details["numeric_feature_count"] + prep_details["categorical_feature_count"]

    with st.sidebar:
        st.header("Model Configuration")
        test_size = st.slider(
            "Test size",
            min_value=0.1,
            max_value=0.4,
            value=float(default_params["test_size"]),
            step=0.05,
        )
        k_max = max(5, total_feature_candidates)
        k_features = st.slider(
            "Top features (SelectKBest)",
            min_value=5,
            max_value=k_max,
            value=min(default_params["k_features"], k_max),
            step=1,
        )
        cv_folds = st.slider("Cross-validation folds", min_value=3, max_value=10, value=default_params["cv_folds"])
        alpha = st.select_slider(
            "Interval significance (alpha)",
            options=[0.01, 0.05, 0.1],
            value=float(default_params["alpha"]),
        )
        random_state = default_params["random_state"]

    summary = run_analysis(
        k_features=k_features,
        test_size=test_size,
        random_state=random_state,
        cv_folds=cv_folds,
        alpha=alpha,
    )

    metrics = summary["train"]["metrics"]
    residual_stats = summary["train"]["residual_stats"]
    cv_results = summary["cv"]

    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Test R²", f"{metrics['r2_test']:.3f}")
    col2.metric("Test RMSE", f"${metrics['rmse_test']:,.0f}")
    col3.metric("Test MAE", f"${metrics['mae_test']:,.0f}")
    st.write(
        f"Cross-validation ({cv_folds} folds) R² = {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}, "
        f"RMSE = ${cv_results['rmse_mean']:,.0f} ± ${cv_results['rmse_std']:,.0f}."
    )

    st.subheader("Prediction Intervals")
    fig = create_prediction_plot(summary["train"]["prediction_df"], alpha)
    st.pyplot(fig)
    plt.close(fig)

    st.write("Sample predictions with intervals:")
    st.dataframe(
        summary["train"]["prediction_df"]
        .rename(
            columns={
                "SalePrice": "Actual",
                "y_pred": "Predicted",
                "pred_lower": "Pred. Lower",
                "pred_upper": "Pred. Upper",
                "conf_lower": "Conf. Lower",
                "conf_upper": "Conf. Upper",
            }
        )
        .head(25)
    )

    st.subheader("Selected Features")
    st.write(
        f"Feature selector retained {summary['parameters']['k_features']} out of "
        f"{summary['train']['total_features']} engineered predictors."
    )
    st.dataframe(summary["train"]["selected_features"])

    st.subheader("Coefficient Summary (Full Dataset Fit)")
    coeff_df = summary["selected_features_full"].copy()
    st.dataframe(coeff_df)

    st.subheader("Residual Diagnostics")
    st.write(
        f"Residual mean {residual_stats['mean']:.2f}, std {residual_stats['std']:.2f}, "
        f"skew {residual_stats['skew']:.2f}, kurtosis {residual_stats['kurtosis']:.2f}."
    )

    report_text = generate_report(summary)
    report_bytes = report_text.encode("utf-8")
    csv_bytes = summary["train"]["prediction_df"].to_csv(index=True).encode("utf-8")

    st.sidebar.download_button(
        label="Download predictions (CSV)",
        data=csv_bytes,
        file_name="predictions_with_intervals.csv",
        mime="text/csv",
    )
    st.sidebar.download_button(
        label="Download report (Markdown)",
        data=report_bytes,
        file_name="analysis_report.md",
        mime="text/markdown",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ames Housing linear regression with feature selection and Streamlit UI",
        add_help=False,
    )
    parser.add_argument("--no-ui", action="store_true", help="Run analysis and save report without launching Streamlit")
    parser.add_argument("--k-features", type=int, default=None, help="Number of top features to keep (SelectKBest)")
    parser.add_argument("--test-size", type=float, default=DEFAULT_PARAMS["test_size"], help="Test set proportion")
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_PARAMS["cv_folds"], help="Cross-validation folds")
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_PARAMS["alpha"],
        help="Significance level for intervals (e.g., 0.05 for 95%)",
    )
    parser.add_argument("--help", action="help", help="Show this help message and exit")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    params = {
        "k_features": args.k_features if args.k_features is not None else DEFAULT_PARAMS["k_features"],
        "test_size": args.test_size,
        "random_state": DEFAULT_PARAMS["random_state"],
        "cv_folds": args.cv_folds,
        "alpha": args.alpha,
    }

    if args.no_ui:
        summary = run_analysis(**params)
        report_text = generate_report(summary)
        save_report_and_plot(summary, report_text)
        print(
            f"Analysis complete. Report saved to {REPORT_PATH} and prediction plot saved to {PLOT_PATH}."
        )
    else:
        run_streamlit_app(params)


if __name__ == "__main__":
    main()
