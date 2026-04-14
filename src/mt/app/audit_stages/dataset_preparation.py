import pandas as pd

from mt.domain.audit import AuditPipelineContext
from mt.app.base_stages.dataset_preparation import BaseDatasetPreparationStage


class AuditDatasetPreparationStage(BaseDatasetPreparationStage):
	name = "audit_dataset_preparation"

	def get_dataset_manifest(self, ctx: AuditPipelineContext):
		return ctx.dataset_manifest

	def after_execute(self, ctx: AuditPipelineContext) -> None:
		if ctx.raw_dataset is None or ctx.dataset is None:
			raise ValueError()

		ctx.raw_context = self._build_raw_context(
			ctx.raw_dataset.sales,
			ctx.raw_dataset.calendar,
			ctx.raw_dataset.sell_prices,
		)

	def _build_raw_context(
		self,
		sales: pd.DataFrame,
		calendar: pd.DataFrame,
		sell_prices: pd.DataFrame,
	) -> dict[str, object]:
		# для восстановления ежедневного масштаба исходного M5 для иллюстраций
		day_columns = [column for column in sales.columns if column.startswith("d_")]
		sample_day_columns = day_columns[:8]
		daily_total_sales = (
			sales.loc[:, day_columns]
			.sum(axis=0)
			.rename_axis("d")
			.reset_index(name="sales_units")
			.merge(calendar.loc[:, ["d", "date"]], on="d", how="left")
		)
		daily_total_sales["date"] = pd.to_datetime(daily_total_sales["date"], utc=False)
		# подсчет sku по категориям помогает показать реальную ширину иерархии до агрегации category
		item_counts_by_category = (
			sales.groupby("cat_id", as_index=False)["item_id"]
			.nunique()
			.rename(columns={"cat_id": "category", "item_id": "item_count"})
		)
		raw_sales_sample_columns = [
			"item_id",
			"dept_id",
			"cat_id",
			"store_id",
			"state_id",
			*sample_day_columns,
		]
		raw_sales_sample = sales.loc[:, raw_sales_sample_columns].head(10).reset_index(drop=True)
		calendar_sample = (
			calendar.loc[:, ["d", "date", "wm_yr_wk"]]
			.head(10)
			.reset_index(drop=True)
		)
		sell_prices_sample = (
			sell_prices.loc[:, ["store_id", "item_id", "wm_yr_wk", "sell_price"]]
			.head(10)
			.reset_index(drop=True)
		)

		return {
			# словарь передается в audit builder как компактный raw-слепок для markdown и графиков
			"raw_item_count": int(sales["item_id"].nunique()),
			"raw_store_count": int(sales["store_id"].nunique()),
			"raw_state_count": int(sales["state_id"].nunique()),
			"raw_department_count": int(sales["dept_id"].nunique()),
			"raw_daily_observations": len(day_columns),
			"daily_total_sales": daily_total_sales.loc[:, ["date", "sales_units"]].sort_values(
				"date").reset_index(drop=True),
			"item_counts_by_category": item_counts_by_category,
			"raw_sales_sample": raw_sales_sample,
			"calendar_sample": calendar_sample,
			"sell_prices_sample": sell_prices_sample,
			"price_available_raw": sell_prices is not None,
			"promo_available_raw": False,
			"price_allowed_for_default_model": False,
			"promo_allowed_for_default_model": False,
			"price_policy_reason": "M5 содержит исторические sell_prices, однако при стандартном недельном прогнозировании нельзя по умолчанию предполагать, что будущая цена продажи известна",
			"promo_policy_reason": "Аудит M5 не предоставляет надежной таблицы с прогнозным планом промоактивностей",
			"calendar_period_start": str(pd.to_datetime(calendar["date"]).min().date()),
			"calendar_period_end": str(pd.to_datetime(calendar["date"]).max().date()),
		}
