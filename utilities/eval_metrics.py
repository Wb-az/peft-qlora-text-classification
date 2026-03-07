from collections import defaultdict
from pathlib import Path

from evaluate import load
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import plotly.express as px


METRICS = ["accuracy", "matthews_correlation", "f1", "precision",
           "recall"]
MACRO_METRICS = {"f1", "precision", "recall"}

_METRIC_CACHE = {}


def _to_numpy_labels(arr_like):
    # Handle tensors from frameworks like torch, including CUDA tensors.
    if hasattr(arr_like, "detach") and hasattr(arr_like, "cpu"):
        arr = arr_like.detach().cpu().numpy()
    elif hasattr(arr_like, "get"):
        arr = arr_like.get()
    else:
        arr = np.asarray(arr_like)

    arr = np.asarray(arr).ravel()
    if arr.size == 0:
        return arr.astype(np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64)
    return arr


def compute_metrics(predictions: np.ndarray, labels: np.ndarray):

    metrics_dict = {}

    for metric in METRICS:
        m = _METRIC_CACHE.get(metric)
        if m is None:
            m = load(metric)
            _METRIC_CACHE[metric] = m

        if metric in MACRO_METRICS:
            res = m.compute(references=labels, predictions=predictions,
                            average="macro")
        else:
            res = m.compute(references=labels, predictions=predictions)

        metrics_dict.update(res)

    return metrics_dict


def class_metrics(eval_pred):

    # Support both eval prediction and tuple unpacking
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)

    metrics = compute_metrics(predictions, labels)

    return metrics


def plot_confusion_matrix(labels, predictions, classes, image_name, outdir, show=True):
    
    outdir = Path(outdir)
    
    outdir.mkdir(parents=True, exist_ok=True)
    save_fig = outdir / f"cm_{image_name}"
  
    confusion_metric = load("confusion_matrix")
    result = confusion_metric.compute(references=labels, predictions=predictions)
    cm = result["confusion_matrix"]

    fig = px.imshow(
        cm,
        text_auto=True,
        x=classes,
        y=classes,
        labels={"x": "Predicted Emotion", "y": "True Emotion", "color": "Count"},
        color_continuous_scale="Blues",
        aspect="equal",
    )

    fig.update_traces(xgap=1, ygap=1)

    fig.update_coloraxes(
        colorbar_x=1.1,         
        colorbar_xanchor="left",
        colorbar_thickness=14,
        colorbar_len=0.9,
    )

    # Colorbar configuration
    fig.update_layout(
        width=620,
        height=620,
        margin=dict(l=60, r=20, t=40, b=60),
    )

    # Prevent extra empty domain spacing
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(constrain="domain")

    fig.write_image(f"{save_fig}.svg", format="svg")
    fig.write_html(f"{save_fig}.html")
    if show:
        fig.show()


def class_report_df(labels, predictions, output_dict=True, m_name=None,
                    save_path='None', mapping_dict=None):

    crep = classification_report(labels, predictions, output_dict=output_dict)

    # Rename class-id keys in classification_report dict safely
    renamed = {}
    for k, v in crep.items():
        if isinstance(k, str) and k.isdigit() and int(k) in mapping_dict:
            renamed[mapping_dict[int(k)]] = v
        else:
            renamed[k] = v

    crep = renamed

    rows = []
    for key, value in crep.items():
        if key == 'accuracy':
            rows.append({
                'Metric': 'accuracy',
                'Precision': '-',
                'Recall': '-',
                'F1-score': round(value, 2),
                'Support': int(crep['macro avg']['support'])
            })
        else:
            rows.append({
                'Metric': key,
                'Precision': round(value['precision'], 2),
                'Recall': round(value['recall'], 2),
                'F1-score': round(value['f1-score'], 2),
                'Support': int(value['support'])
            })

    df = pd.DataFrame(rows)
    if save_path is not None:
        file_name = Path(save_path) / f"{m_name}_class_report.csv"

        df.to_csv(file_name, index=False)
    return df


def bootstrap_metrics(labels: np.ndarray, predictions: np.ndarray,
                      rounds=1000, seed=1234):

    labels = _to_numpy_labels(labels)
    predictions = _to_numpy_labels(predictions)

    if labels.shape[0] != predictions.shape[0]:
        raise ValueError(f"predictions and labels must have same length."
                         f" Got {len(predictions)} and {len(labels)}")
    if rounds <= 0:
        raise ValueError(f"rounds must be > 0. Got {rounds}")

    out = defaultdict(list)
    size = len(labels)
    classes = np.unique(np.concatenate([labels, predictions]))
    n_classes = len(classes)
    labels_encoded = np.searchsorted(classes, labels)
    predictions_encoded = np.searchsorted(classes, predictions)

    rng = np.random.default_rng(seed=seed)
    for _ in range(rounds):
        idx = rng.integers(low=0,  high=size, size=size)
        y_true = labels_encoded[idx]
        y_pred = predictions_encoded[idx]

        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        np.add.at(cm, (y_true, y_pred), 1)

        tp = np.diag(cm).astype(np.float64)
        t_sum = cm.sum(axis=1).astype(np.float64)
        p_sum = cm.sum(axis=0).astype(np.float64)
        n = float(cm.sum())
        n_correct = float(tp.sum())

        with np.errstate(divide='ignore', invalid='ignore'):
            precision_per_class = np.divide(tp, p_sum, out=np.zeros_like(tp), where=p_sum != 0)
            recall_per_class = np.divide(tp, t_sum, out=np.zeros_like(tp), where=t_sum != 0)
            f1_per_class = np.divide(
                2.0 * precision_per_class * recall_per_class,
                precision_per_class + recall_per_class,
                out=np.zeros_like(tp),
                where=(precision_per_class + recall_per_class) != 0,
            )

        cov_ytyp = n_correct * n - np.dot(t_sum, p_sum)
        cov_ytyt = n * n - np.dot(t_sum, t_sum)
        cov_ypyp = n * n - np.dot(p_sum, p_sum)
        denom = np.sqrt(cov_ytyt * cov_ypyp)
        mcc = cov_ytyp / denom if denom > 0 else 0.0

        out["accuracy"].append(n_correct / n if n > 0 else 0.0)
        out["precision"].append(float(np.mean(precision_per_class)))
        out["recall"].append(float(np.mean(recall_per_class)))
        out["f1"].append(float(np.mean(f1_per_class)))
        out["matthews_correlation"].append(float(mcc))

    return out


def confidence_intervals(scores, alpha=0.05):
    lower_p = alpha / 2.0
    upper_p = 1.0 - alpha / 2.0
    lower, upper = np.quantile(scores, [lower_p, upper_p])
    return lower, upper


def metrics_summary(predictions: np.ndarray, labels: np.ndarray,
                    rounds: int = 1000, alpha: float = 0.05, model_name: str = None,
                    seed: int = 1234):

    res = defaultdict()

    metrics = bootstrap_metrics(labels, predictions, rounds=rounds,
                              seed=seed)

    for metric, value in metrics.items():
        med = np.median(value)
        avg = np.mean(value)
        low, high = confidence_intervals(value, alpha)

        res[metric] = {'average': avg, 'median': med,
                     'ci_lower': low, 'ci_upper': high}

    res_df = pd.DataFrame.from_dict(res, orient='index')
    res_df.index = pd.MultiIndex.from_product([[model_name], res_df.index], names=['model', 'metric'])

    return res_df
    

def plot_loss(history_log: dict, model_name: str, save_dir: Path = None) -> None:

    log_dict = defaultdict(list)
    for log in history_log:
        for k in log.keys():
            if  k == 'loss':
                log_dict['loss'].append(log['loss'])
                log_dict['step'].append(log['step'])

    df = pd.DataFrame(log_dict)
    fig = px.line(df, x='step', y='loss', width=500, height=400,
                  title=f'Evaluation loss {model_name}', template='simple_white')
    fig.write_image(f'{save_dir}/loss_{model_name}.svg', format="svg")
    fig.show()