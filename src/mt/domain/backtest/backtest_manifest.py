from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class BacktestManifest:
	"""Настройки rolling window backtesting"""

	horizon_start: int = 1
	horizon_end: int = 8
	min_train_weeks: int = 52
	step_weeks: int = 1
	shared_origin_grid: bool = True
	bootstrap_samples: int = 300
	holdout_tail_weeks: int | str | None = "auto"

	def __post_init__(self) -> None:
		if self.horizon_start < 1 or self.horizon_end > 8 or self.horizon_start > self.horizon_end:
			raise ValueError()
		if self.min_train_weeks < 8 or self.step_weeks < 1:
			raise ValueError()
		if self.bootstrap_samples < 1:
			raise ValueError()
		if self.holdout_tail_weeks in {None, "auto"}:
			return
		if not isinstance(self.holdout_tail_weeks, int):
			raise ValueError("backtest.holdout_tail_weeks must be int, auto or null")
		if self.holdout_tail_weeks < 0 or self.holdout_tail_weeks > 8:
			raise ValueError("backtest.holdout_tail_weeks must stay within 0..8")

	def resolve_holdout_tail_weeks(self) -> int:
		if self.holdout_tail_weeks == "auto":
			return self.horizon_end
		if self.holdout_tail_weeks is None:
			return 0
		return int(self.holdout_tail_weeks)
