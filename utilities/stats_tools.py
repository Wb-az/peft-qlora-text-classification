from pathlib import Path
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as ss


__all__ = ['friedman_comparison', 'model_ranking', 'plot_posthoc', 'plot_critical_diff',
           'friedman_nemenyi_analysis', 'class_model_comparison']


def friedman_comparison(data: pd.DataFrame) -> tuple[np.float64, np.float64, pd.DataFrame | None]:
    """
    :param data: comparison results
    """
    data_values = data.values
    stat, pvalue = ss.friedmanchisquare(*data_values)

    if pvalue < 0.05:
        pc = sp.posthoc_nemenyi_friedman(data.T)
        pc.columns = data.columns
        pc.index = data.columns
        return stat, pvalue, pc

    else:
        print('Fail to reject the null hypothesis, pvalue: {:3f}'.format(pvalue))
        return stat, pvalue, None


def model_ranking(df: pd.DataFrame):
    """
    :param df: data to rank
    :rtype: pd.Series
    """
    ranks = df.rename_axis('cv_fold').melt(
        var_name='estimator',
        value_name='value',
        ignore_index=False).reset_index()

    avg_rank = ranks.groupby('cv_fold').value.rank(pct=True).groupby(ranks.estimator).mean()

    return avg_rank


def plot_posthoc(pc: pd.DataFrame, fig_name: str, outdir: str,
                 plot_size: (tuple | list) | None = (7, 6),
                 show: bool = True) -> None:
    """
    :param pc: posthoc test result
    :param plot_size: size of figure
    :type plot_size: tuple(int, int) | list[int, int] | plot_size is None
    :param fig_name: name of figure
    :param outdir: directory to save figure
    :param show: display or not the figure
    """

    # Setup figure
    if plot_size is not None and len(plot_size) != 2:
        raise ValueError(f"plot_size must be 2D (width, height), got {len(plot_size)} element(s).")

    fig = plt.figure(figsize=plot_size) if plot_size is not None else plt.figure()

    plt.subplots_adjust(left=0.2, right=0.75, top=0.9, bottom=0.1)

    # Prepare the path
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / f"{fig_name}.svg"

    # Plot design
    cmap = ['#ffffff', '#FADCDC', '#08306b', '#4292c6', '#c6dbef']
    heatmap_args = {'cmap': cmap, 'linewidths': 0.2, 'linecolor': 'black',
                    'clip_on': False, 'square': True,
                    'cbar_ax_bbox': [0.80, 0.33, 0.04, 0.3]}

    # Create plot
    hax, _ = sp.sign_plot(pc, **heatmap_args, labels=True)
    hax.set_yticklabels(pc.columns, rotation=0)

    # Save and clean up
    plt.savefig(save_path, format='svg', dpi=96)

    if show:
        plt.show()
        plt.close(fig)


def plot_critical_diff(ranks: pd.Series, posthoc_test: pd.DataFrame, outdir: str,
                       fig_name: str, plot_size: tuple[int, int] | list[int] | np.ndarray| None = (6, 4), 
                       show: bool = True) -> None:
    """
    :param ranks: Ranks from model ranking
    :param posthoc_test: posthoc test result
    :param fig_name: name of figure
    :param outdir:  directory to save figure
    :param plot_size: size of figure
    :param show: display or not the figure
    """

    # Setup figure
    if plot_size is not None and len(plot_size) != 2:
        raise ValueError(f"plot_size must be 2D (width, height), got {len(plot_size)} elements.")

    fig = plt.figure(figsize=plot_size) if plot_size is not None else plt.figure()

    plt.title('Critical difference diagram of average score ranks')

    # Prepare the path
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = Path(outdir) / f"{fig_name}.svg"

    # Create plot
    sp.critical_difference_diagram(ranks, posthoc_test)

    # Save and clean up
    plt.savefig(save_path, format='svg', dpi=96)

    if show:
        plt.show()
        plt.close(fig)


def friedman_nemenyi_analysis(df: pd.DataFrame, output_dir: str = "results",
                              **kwargs) -> None:
    """
    :param df: data to compare
    :param output_dir: directory to save the plots
    :param kwargs: plot parameters
    :return: None
    """

    print(f"--- Processing {len(df.columns)} models ---\n")

    ranks = model_ranking(df)
    _, p_val, pc = friedman_comparison(df)

    if pc is not None:
        print("--- Friedman-Nemenyi statistical comparison - significance_heatmap ---\n")
        plot_posthoc(pc, kwargs["hmap_name"], output_dir, kwargs["hm_size"])
        print("")
        print("--- Critical difference Plot ---\n")
        plot_critical_diff(ranks, pc, output_dir, kwargs["cd_diagram"], kwargs["cd_size"])
        print(f"Significant difference (p={p_val:.4f}). Plots saved to {output_dir}")
    else:
        print(f"Non-significant results (p={p_val:.4f}). No plots generated.")


def cochrans_q_test(binary_results):

    q_result = cochrans_q(binary_results)

    print(f"Cochran's Q Statistic: {q_result.statistic:.4f}")
    print(f"P-value: {q_result.pvalue:.4f}")

    # Interpretation
    if q_result.pvalue < 0.05:
        print("\nResult is Significant: At least one model performs differently.")
        print("Proceed to pairwise McNemar tests to find the best model(s).")
    else:
        print("\nNo significant difference found between the models.")

    return q_result


def compute_mcnemar(a, b):

    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)

    n11 = np.sum((a == 1) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0))  # A correct, B wrong
    n01 = np.sum((a == 0) & (b == 1))  # A wrong, B correct
    n00 = np.sum((a == 0) & (b == 0))

    table = np.array([[n11, n10],
                      [n01, n00]], dtype=int)

    mc = mcnemar(table, exact=False, correction=True)
    return mc


def mcnemar_multimodel_comparison(df) -> pd.DataFrame:

    cols = list(df.columns)
    mn = np.full((len(cols), len(cols)), np.nan,  dtype=float)
    np.fill_diagonal(mn, 1.0)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            res = compute_mcnemar(df[cols[i]], df[cols[j]])
            mn[i, j] = res.pvalue
            mn[j, i] = res.pvalue

    return pd.DataFrame(mn, index=df.columns, columns=df.columns)


def accuracy_diff_matrix(binary_results: pd.DataFrame) -> pd.DataFrame:
    
    binary = binary_results.to_numpy().astype(int)  # N x k
    n, k = binary.shape
    ones = binary
    zeros = 1 - binary

    # n10[i,j] = counts where model i=1 and model j=0
    n10 = ones.T @ zeros
    n01 = zeros.T @ ones

    acc_diff = (n10 - n01) / n
    np.fill_diagonal(acc_diff, 0.0)

    return pd.DataFrame(acc_diff, index=binary_results.columns, columns=binary_results.columns)


def holm_on_mcnemar_matrix(pmat: pd.DataFrame) -> pd.DataFrame:
    
    prob = pmat.to_numpy().copy()
    iu = np.triu_indices_from(prob, k=1)
    p = prob[iu]

    _, p_adj, _, _ = multipletests(p, method='holm')

    prob2 = prob.copy()
    prob2[iu] = p_adj
    prob2[(iu[1], iu[0])] = p_adj
    np.fill_diagonal(prob2, 1.0)

    return pd.DataFrame(prob2, index=pmat.index, columns=pmat.columns)


def class_model_comparison(binary_results : pd.DataFrame, plot=True, **kwargs):
    
    q_res = cochrans_q_test(binary_results)

    if q_res.pvalue >= 0.05:
        return None

    else:
        print('Initiating posthoc McNemar pair-wise comparison')
        comparison = mcnemar_multimodel_comparison(binary_results)
        correct_comp = holm_on_mcnemar_matrix(comparison)

        # Direction - effect size (accuracy difference)
        acc_dir = accuracy_diff_matrix(binary_results)

        if plot:
            plot_posthoc(comparison, fig_name='mc_nemar_test',
                         outdir=kwargs['outdir'],
                         plot_size=kwargs['size'], show=kwargs['show'])

        return correct_comp, acc_dir