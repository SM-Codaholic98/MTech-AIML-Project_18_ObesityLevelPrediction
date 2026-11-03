"""Run model evaluation, SHAP comparison, and export reproducible results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from .preprocessing import load_dataset, prepare_data
    from .model import build_models, fit_models, calculate_grouped_shap_importance
except ImportError:  # Allows running this file directly as well.
    from preprocessing import load_dataset, prepare_data
    from model import build_models, fit_models, calculate_grouped_shap_importance


def evaluate_models(models, splits):
    partitions = {
        "Train": (splits.X_train, splits.y_train),
        "Validation": (splits.X_val, splits.y_val),
        "Test": (splits.X_test, splits.y_test),
    }
    rows = []
    for model_name, model in models.items():
        for split_name, (X_part, y_part) in partitions.items():
            predictions = model.predict(X_part)
            rows.append({
                "Model": model_name,
                "Dataset": split_name,
                "Accuracy": accuracy_score(y_part, predictions),
                "Macro_F1": f1_score(y_part, predictions, average="macro"),
                "Weighted_F1": f1_score(y_part, predictions, average="weighted"),
                "Macro_Precision": precision_score(
                    y_part, predictions, average="macro", zero_division=0
                ),
                "Macro_Recall": recall_score(
                    y_part, predictions, average="macro", zero_division=0
                ),
                "Samples": len(y_part),
            })
    return pd.DataFrame(rows)


def compare_explanations(importance_by_model, top_n=5):
    names = list(importance_by_model)
    if len(names) < 2:
        raise ValueError("At least two fitted classifiers are required for comparison.")

    first_name, second_name = names[:2]
    first = importance_by_model[first_name]
    second = importance_by_model[second_name]
    all_features = sorted(set(first.index) | set(second.index))
    first_values = first.reindex(all_features).fillna(0)
    second_values = second.reindex(all_features).fillna(0)

    first_top = set(first.nlargest(top_n).index)
    second_top = set(second.nlargest(top_n).index)
    common = sorted(first_top & second_top)
    overlap = len(common) / top_n * 100

    first_ranks = first_values.rank(ascending=False, method="average")
    second_ranks = second_values.rank(ascending=False, method="average")
    correlation, p_value = spearmanr(first_ranks, second_ranks)

    comparison = pd.DataFrame([{
        "Model_1": first_name,
        "Model_2": second_name,
        "Top_N": top_n,
        "Model_1_Top_Features": "; ".join(first.nlargest(top_n).index),
        "Model_2_Top_Features": "; ".join(second.nlargest(top_n).index),
        "Common_Top_Features": "; ".join(common),
        "Top_N_Overlap_Percent": overlap,
        "Spearman_Rank_Correlation": correlation,
        "Spearman_P_Value": p_value,
    }])

    feature_comparison = pd.DataFrame({
        "Feature": all_features,
        first_name: first_values.values,
        second_name: second_values.values,
        f"{first_name}_Rank": first_ranks.values,
        f"{second_name}_Rank": second_ranks.values,
    })
    return comparison, feature_comparison


def save_figures(feature_comparison, model_names, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    first_name, second_name = model_names[:2]
    plot_df = feature_comparison.set_index("Feature")[[first_name, second_name]]
    plot_df = plot_df.sort_values(first_name, ascending=True)
    ax = plot_df.plot(kind="barh", figsize=(10, 8), width=0.8)
    ax.set_title("Grouped SHAP Feature Importance Comparison")
    ax.set_xlabel("Mean Absolute Grouped SHAP Value")
    ax.set_ylabel("Original Feature")
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(figures_dir / "grouped_shap_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def run(data_path: Path, results_dir: Path, random_state: int = 42):
    results_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)
    splits = prepare_data(df, random_state=random_state)
    models = fit_models(build_models(random_state), splits.X_train, splits.y_train)

    performance = evaluate_models(models, splits)
    performance.to_csv(results_dir / "results.csv", index=False, float_format="%.6f")

    importance = {
        name: calculate_grouped_shap_importance(model, splits.X_test)
        for name, model in models.items()
    }
    comparison, feature_comparison = compare_explanations(importance)
    comparison.to_csv(results_dir / "comparison_table.csv", index=False, float_format="%.6f")
    feature_comparison.to_csv(
        results_dir / "feature_importance_comparison.csv", index=False, float_format="%.8f"
    )
    save_figures(feature_comparison, list(models), results_dir / "figures")

    print("Completed evaluation.")
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Train/validation/test sizes: {len(splits.y_train)}/{len(splits.y_val)}/{len(splits.y_test)}")
    print("\nTest performance:")
    print(performance[performance["Dataset"] == "Test"].to_string(index=False))
    print("\nExplanation similarity:")
    print(comparison.to_string(index=False))
    print(f"\nResults saved to: {results_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ObesityDataSet_raw_and_data_sinthetic.csv"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    run(args.data, args.results, args.random_state)


if __name__ == "__main__":
    main()
