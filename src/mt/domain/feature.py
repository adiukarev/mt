from dataclasses import dataclass


@dataclass(slots=True)
class FeatureSpec:
	"""Описание признака и его допустимости на дату прогноза"""

	# Имя признака
	name: str
	# Группа признаков
	group: str
	# Источник данных для признака
	source: str
	# Краткое описание расчета
	calculation: str
	# Класс ковариаты: observed / known_in_advance / future_unknown (тип ковариаты по правилу доступности во времени)
	# observed - наблюдается только по мере течения времени
	# known_in_advance - известна заранее для будущих дат
	# future_unknown - в будущем неизвестна и запрещена без доказанной доступности.
	covariate_class: str
	# Ожидаемый механизм влияния на продажи
	expected_effect_mechanism: str
	# Доступен ли признак на дату прогноза
	availability_at_forecast_time: bool
	# Включен ли признак в текущий набор
	enabled: bool = True
	# Причина отключения признака
	reason_if_disabled: str = ""
