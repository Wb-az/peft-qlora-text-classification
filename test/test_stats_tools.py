import shutil
from pathlib import Path
from utilities.stats_tools import *
import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope="module")
def sample_df():
    data_array = np.array([
        [8.82, 11.8, 10.37, 12.08],
        [8.92, 9.58, 10.59, 11.89],
        [8.27, 11.46, 10.24, 11.6],
        [8.83, 13.25, 8.33, 11.51],
    ])
    return pd.DataFrame(data_array.T, columns=["A", "B", "C", "D"])


@pytest.fixture(scope="session")
def debug_plot_dir():
    p = Path("test/debug_plots")
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture(scope="module")
def shared_friedman_comparison(sample_df):
    return friedman_comparison(sample_df)


@pytest.fixture(scope="module")
def shared_ranks(sample_df):
    return model_ranking(sample_df)


def test_friedman_comparison(shared_friedman_comparison):

    assert isinstance(shared_friedman_comparison, tuple)
    assert len(shared_friedman_comparison) == 3

    stat, pvalue, pc = shared_friedman_comparison

    assert 0.0 <= pvalue <= 1.0
    assert isinstance(pc, pd.DataFrame | None)

    if pc is not None:

        rows, cols = pc.shape
        assert rows == cols, f"Post-hoc matrix should be square, got {rows}x{cols}"
        assert list(pc.columns) == list(pc.index)


def test_model_ranking(shared_ranks, sample_df):

    assert isinstance(shared_ranks, pd.Series)
    assert len(shared_ranks) == sample_df.shape[1]
    assert set(shared_ranks.index) == set(sample_df.columns)


@pytest.mark.parametrize("size, expected_error", [(None, None), ((10, 5), None),
                                                  ([3.5], ValueError)])
def test_plot_posthoc(shared_friedman_comparison, debug_plot_dir, size, expected_error):

    # Unpack friedman results
    _, _, pc = shared_friedman_comparison

    # Evaluate incorrect plot size
    if expected_error:
        with pytest.raises(expected_error):
            plot_posthoc(pc, "dummy_name", str(debug_plot_dir), size)
        return

    # Figure name
    label = "none" if size is None else f"{size[0]}x{size[1]}"
    figname = f"posthoc_test_plot_{label}"

    # Create plot
    plot_posthoc(pc, fig_name=figname, outdir=debug_plot_dir,
                 plot_size=size)

    output_file = debug_plot_dir / f'{figname}.svg'
    assert output_file.is_file()
    assert output_file.stat().st_size > 0


@pytest.mark.parametrize("size, expected_error", [((5, 4), None), (None, None),
    ([0], ValueError)])
def test_plot_critical_diff(shared_ranks, shared_friedman_comparison, debug_plot_dir, size, expected_error):

    # Unpack friedman results
    _, _, pc = shared_friedman_comparison

    # Evaluate incorrect plot size
    if expected_error:
        with pytest.raises(expected_error):
            plot_critical_diff(shared_ranks, pc, str(debug_plot_dir),
                               "err_test", size)
        return

    # Figure name
    label = "none" if size is None else f"{size[0]}x{size[1]}"
    figname = f"cd_plot_{label}"

    # Create plot
    plot_critical_diff(shared_ranks, posthoc_test=pc, outdir=str(debug_plot_dir),
                       fig_name=figname, plot_size=size)

    # Assertions
    output_file = debug_plot_dir / f"{figname}.svg"
    assert output_file.exists(), f"File {figname}.svg was not created."
    assert output_file.stat().st_size > 0, "SVG file is empty."

