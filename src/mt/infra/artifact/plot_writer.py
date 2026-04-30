from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mt.infra.artifact.plot_labels import set_axis_labels


def save_histogram(frame: pd.DataFrame, column: str, title: str, output_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(8, 4))
	frame[column].dropna().plot(kind="hist", bins=20, ax=ax, title=title)
	set_axis_labels(ax, xlabel=column, ylabel="count")
	save_figure(fig, output_path)


def save_figure(figure: plt.Figure, path: Path, dpi: int | None = None) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	figure.tight_layout()
	if dpi is None:
		figure.savefig(path)
	else:
		figure.savefig(path, dpi=dpi)
	plt.close(figure)
