import pandas as pd
from pathlib import Path

import torch
from utilities.eval_metrics import (
    class_report_df,
    metrics_summary,
    plot_confusion_matrix,
)
from utilities.hf_pipeline import inf_predictions
from utilities.stats_tools import class_model_comparison

DEFAULT_MAPPING = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}


def build_runtime_args(data_dir: str | Path, device: str) -> dict:
    """
    Build all required kwargs for inference, metrics, and model comparison.
    :param data_dir: Path to the data directory
    :param device: spcufied whether to use CPU or GPU
    :return: A dictionary of all required kwargs
    :rtype: dict
    """
    """Build all required kwargs for inference, metrics, and model comparison."""

    plot_dir = Path("results/plots")
    metrics_dir = Path("results/metrics")
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    mapping_dict = DEFAULT_MAPPING.copy()
    return {
        "data_dir": data_dir,
        "plot_dir": plot_dir,
        "metrics_dir": metrics_dir,
        "mapping_dict": mapping_dict,
        "num_labels": len(mapping_dict),
        "id2label": mapping_dict,
        "label2id": {v: k for k, v in mapping_dict.items()},
        "device": device,
        "quantized": False,
        "max_length": 128,
        "batch": 32,
        "rounds": 1000,
        "outdir": str(plot_dir),
        "size": None,
        "show": True,
        "model_name": "qlora_model",
        "models_list": [],
        "seed": 1234,
    }


def run_test(adapter_dir, x_test, y_test, **args):
    out, model, m_name = inf_predictions(adapter_dir, x_test, y_test, **args)

    if "/" in m_name:
        m_name = m_name.split("/")[-1]
    lbs, preds = out[0], out[1]

    cls_report = class_report_df(lbs, preds, mapping_dict=args['mapping_dict'],
                                 output_dict=True, m_name=m_name,
                                 save_path=args['metrics_dir'])

    print(cls_report)

    classes = list(args['mapping_dict'].values())

    plot_confusion_matrix(
        lbs,
        preds,
        classes=classes,
        image_name=f"cm_{m_name}",
        outdir=args["plot_dir"],
    )

    print("")
    bootstrap_metrics = metrics_summary(
        predictions=preds,
        labels=lbs,
        rounds=args["rounds"],
        model_name=m_name,
        seed=11,
    )
    print(bootstrap_metrics)
    # Save predictions for comparison for individual model evaluations
    bootstrap_metrics.to_csv(args['metrics_dir'] / f"bs-qlora-{m_name}.csv", index=False)

    return lbs, preds, m_name


def ensure_list(obj):
    """Standardizes input to be a list."""
    return obj if isinstance(obj, list) else [obj]


if __name__ == "__main__":

    # Can be one of: Path("local/dir"), "hf/repo", or a [list] of either.
    adapt_dir = ["Wb-az/peft-roberta-base", "Wb-az/peft-opt-350m", "Wb-az/peft-modernbert-base"]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = build_runtime_args(data_dir=Path("dataset"), device=device)

    # Data Loading
    test = pd.read_csv(Path(args["data_dir"]) / "test_dataset.csv")
    x_test, y_test = test["text"].values, test["label"].values

    # Test execution
    binary_results = {}

    for adapter in ensure_list(adapt_dir):
        print(f"\n>>> Running Evaluation: {adapter}")

        # The run_test extracts 'm_name' internally
        lbs, preds, m_name = run_test(adapter, x_test, y_test, **args)

        # Store results for the comparison table
        binary_results[m_name] = (preds == lbs).astype(int)

    # Triggers comparison if more than one adapter is provided
    if len(binary_results) > 1:
        print("\n--- Running Group Comparison ---")
        binary_res_df = pd.DataFrame(binary_results).astype(int)
        comparison = class_model_comparison(binary_res_df, plot=True, **args)

        if comparison:
            print("\nComparison Matrix:\n", comparison[0])
    else:
        print("\nSingle model evaluation complete.")