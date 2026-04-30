import logging
import os

THIRD_PARTY_LOGGER_NAMES = (
	"lightning",
	"lightning.pytorch",
	"lightning_fabric",
	"lightning_utilities",
	"pytorch_lightning",
	"neuralforecast",
)


def configure_third_party_loggers() -> None:
	level_name = os.getenv("MT_THIRD_PARTY_LOG_LEVEL", "WARNING").strip().upper()
	level = getattr(logging, level_name, logging.WARNING)
	if not isinstance(level, int):
		level = logging.WARNING

	for logger_name in THIRD_PARTY_LOGGER_NAMES:
		logger = logging.getLogger(logger_name)
		logger.setLevel(level)
		logger.propagate = True


class ThirdPartyNoiseFilter(logging.Filter):
	def filter(self, record: logging.LogRecord) -> bool:
		level_name = os.getenv("MT_THIRD_PARTY_LOG_LEVEL", "WARNING").strip().upper()
		threshold = getattr(logging, level_name, logging.WARNING)
		if not isinstance(threshold, int):
			threshold = logging.WARNING

		name = (record.name or "").lower()
		is_third_party = any(name.startswith(prefix) for prefix in THIRD_PARTY_LOGGER_NAMES)
		if not is_third_party:
			return True
		return record.levelno >= threshold
