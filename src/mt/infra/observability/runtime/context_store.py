from contextvars import ContextVar, Token

from mt.domain.observability.observability import ObservabilityContext

_CURRENT_OBSERVABILITY: ContextVar[ObservabilityContext | None] = ContextVar(
	"mt_current_observability",
	default=None,
)


def bind_observability(ctx: ObservabilityContext | None) -> Token[ObservabilityContext | None]:
	return _CURRENT_OBSERVABILITY.set(ctx)


def reset_observability(token: Token[ObservabilityContext | None]) -> None:
	_CURRENT_OBSERVABILITY.reset(token)


def get_observability() -> ObservabilityContext | None:
	return _CURRENT_OBSERVABILITY.get()
