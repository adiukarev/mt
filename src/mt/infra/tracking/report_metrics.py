import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class ParsedReportPayload:
	metrics: dict[str, float]
	params: dict[str, str]


_HEADING_RE = re.compile(r"^(#+)\s+(.*\S)\s*$")
_KEY_VALUE_RE = re.compile(r"^(?P<key>[^:=]+?)\s*(?P<sep>:|=)\s*(?P<value>.+)$")
_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_RATIO_RE = re.compile(r"^(?P<numerator>-?\d+(?:\.\d+)?)\s*/\s*(?P<denominator>-?\d+(?:\.\d+)?)$")
_VARIABLES_IN_BRACKETS_RE = re.compile(r"^(?P<label>.*?)\s*\[(?P<vars>[^\]]+)\]\s*$")


def parse_report_file(path: str | Path) -> ParsedReportPayload:
	report_path = Path(path)
	if not report_path.exists():
		return ParsedReportPayload(metrics={}, params={})
	return parse_report_lines(report_path.read_text(encoding="utf-8").splitlines())


def parse_report_lines(lines: list[str]) -> ParsedReportPayload:
	metrics: dict[str, float] = {}
	params: dict[str, str] = {}
	section_parts: list[str] = []
	table_headers: list[str] | None = None

	for raw_line in lines:
		line = raw_line.strip()
		if not line:
			table_headers = None
			continue
		if heading_match := _HEADING_RE.match(line):
			level = len(heading_match.group(1))
			title = _slugify(heading_match.group(2))
			section_parts = section_parts[:max(level - 1, 0)]
			section_parts.append(title)
			table_headers = None
			continue
		if line.startswith("|"):
			table_headers = _process_table_line(
				line=line,
				section_parts=section_parts,
				table_headers=table_headers,
				metrics=metrics,
				params=params,
			)
			continue
		if not line.startswith("- "):
			continue
		_process_bullet_line(
			line=line[2:].strip(),
			section_parts=section_parts,
			metrics=metrics,
			params=params,
		)

	return ParsedReportPayload(metrics=metrics, params=params)


def _process_table_line(
	line: str,
	section_parts: list[str],
	table_headers: list[str] | None,
	metrics: dict[str, float],
	params: dict[str, str],
) -> list[str] | None:
	cells = _parse_table_cells(line)
	if not cells:
		return table_headers
	if _is_separator_row(cells):
		return table_headers
	if table_headers is None:
		return cells
	_process_table_row(
		cells=cells,
		headers=table_headers,
		section_parts=section_parts,
		metrics=metrics,
		params=params,
	)
	return table_headers


def _process_table_row(
	cells: list[str],
	headers: list[str],
	section_parts: list[str],
	metrics: dict[str, float],
	params: dict[str, str],
) -> None:
	row_index = _next_table_row_index(params, metrics, section_parts)
	base_prefix = ".".join(["report", *section_parts, f"row_{row_index}"]) if section_parts else f"report.row_{row_index}"

	for header, cell in zip(headers, cells, strict=False):
		header_key = _normalize_table_header(header)
		if header_key == "analysis_block":
			params[f"{base_prefix}.analysis_block"] = cell
			continue
		if header_key == "what_we_check":
			_process_what_we_check_cell(
				cell=cell,
				base_prefix=base_prefix,
				analysis_block=params.get(f"{base_prefix}.analysis_block", ""),
				metrics=metrics,
				params=params,
			)
			continue
		_process_table_cell(
			cell=cell,
			prefix=f"{base_prefix}.{header_key}",
			metrics=metrics,
			params=params,
		)


def _process_table_cell(
	cell: str,
	prefix: str,
	metrics: dict[str, float],
	params: dict[str, str],
) -> None:
	parts = [part.strip() for part in cell.split(";") if part.strip()]
	if not parts:
		return
	if len(parts) == 1 and _KEY_VALUE_RE.match(parts[0]) is None:
		params[prefix] = parts[0]
		return
	for index, part in enumerate(parts):
		if _KEY_VALUE_RE.match(part):
			_process_part(
				part=part,
				prefix=prefix,
				position=index,
				metrics=metrics,
				params=params,
			)
			continue
		params[f"{prefix}.line_{index}"] = part


def _process_what_we_check_cell(
	cell: str,
	base_prefix: str,
	analysis_block: str,
	metrics: dict[str, float],
	params: dict[str, str],
) -> None:
	variable_names = _extract_analysis_variables(analysis_block)
	parts = [part.strip() for part in cell.split(";") if part.strip()]
	prefix = f"{base_prefix}.what_we_check"
	if variable_names and len(variable_names) == len(parts):
		for variable_name, part in zip(variable_names, parts, strict=False):
			_store_table_value(
				prefix=f"{prefix}.{_slugify(variable_name)}",
				value=part,
				metrics=metrics,
				params=params,
			)
		return
	_process_table_cell(
		cell=cell,
		prefix=prefix,
		metrics=metrics,
		params=params,
	)


def _parse_table_cells(line: str) -> list[str]:
	if not line.startswith("|") or not line.endswith("|"):
		return []
	return [cell.strip() for cell in line.strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
	return all(re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells)


def _next_table_row_index(
	params: dict[str, str],
	metrics: dict[str, float],
	section_parts: list[str],
) -> int:
	base_prefix = ".".join(["report", *section_parts]) if section_parts else "report"
	prefix = f"{base_prefix}.row_"
	known_indices: set[int] = set()
	for key in [*params.keys(), *metrics.keys()]:
		if not key.startswith(prefix):
			continue
		suffix = key[len(prefix):]
		row_token = suffix.split(".", maxsplit=1)[0]
		if row_token.isdigit():
			known_indices.add(int(row_token))
	return 0 if not known_indices else max(known_indices) + 1


def _normalize_table_header(header: str) -> str:
	header_map = {
		"блок анализа": "analysis_block",
		"что проверяем": "what_we_check",
		"значение": "value",
		"что делать в модели": "what_to_do_in_model",
		"feature formula policy": "feature_formula_policy",
	}
	return header_map.get(header.strip().lower(), _slugify(header))


def _extract_analysis_variables(analysis_block: str) -> list[str]:
	match = _VARIABLES_IN_BRACKETS_RE.match(analysis_block.strip())
	if not match:
		return []
	return [item.strip() for item in match.group("vars").split(",") if item.strip()]


def _process_bullet_line(
	line: str,
	section_parts: list[str],
	metrics: dict[str, float],
	params: dict[str, str],
) -> None:
	parts = [part.strip() for part in line.split("|") if part.strip()]
	if not parts:
		return
	base_prefix = ".".join(["report", *section_parts]) if section_parts else "report"
	anchor = _anchor_from_part(parts[0])
	prefix = ".".join(part for part in (base_prefix, anchor) if part)
	for index, part in enumerate(parts):
		_process_part(
			part=part,
			prefix=prefix if index > 0 else base_prefix,
			position=index,
			metrics=metrics,
			params=params,
		)


def _process_part(
	part: str,
	prefix: str,
	position: int,
	metrics: dict[str, float],
	params: dict[str, str],
) -> None:
	match = _KEY_VALUE_RE.match(part)
	if not match:
		params[f"{prefix}.line_{position}"] = part
		return
	key = _slugify(match.group("key"))
	value = match.group("value").strip()
	qualified_key = ".".join(filter(None, [prefix, key]))
	numeric_value = _parse_number(value)
	if numeric_value is not None:
		metrics[qualified_key] = numeric_value
		return
	if ratio_match := _RATIO_RE.match(value):
		numerator = float(ratio_match.group("numerator"))
		denominator = float(ratio_match.group("denominator"))
		metrics[f"{qualified_key}.numerator"] = numerator
		metrics[f"{qualified_key}.denominator"] = denominator
		if denominator != 0:
			metrics[f"{qualified_key}.share"] = numerator / denominator
		return
	params[qualified_key] = value


def _store_table_value(
	prefix: str,
	value: str,
	metrics: dict[str, float],
	params: dict[str, str],
) -> None:
	numeric_value = _parse_number(value)
	if numeric_value is not None:
		metrics[prefix] = numeric_value
		return
	if ratio_match := _RATIO_RE.match(value):
		numerator = float(ratio_match.group("numerator"))
		denominator = float(ratio_match.group("denominator"))
		metrics[f"{prefix}.numerator"] = numerator
		metrics[f"{prefix}.denominator"] = denominator
		if denominator != 0:
			metrics[f"{prefix}.share"] = numerator / denominator
		return
	params[prefix] = value


def _anchor_from_part(part: str) -> str:
	match = _KEY_VALUE_RE.match(part)
	if not match:
		return ""
	key = _slugify(match.group("key"))
	if key:
		return key
	return ""


def _parse_number(value: str) -> float | None:
	candidate = value.strip()
	if not _NUMBER_RE.match(candidate):
		return None
	return float(candidate)


def _slugify(value: str) -> str:
	slug = value.strip().lower()
	slug = re.sub(r"[^\w\s.-]", "", slug, flags=re.UNICODE)
	slug = re.sub(r"[\s./-]+", "_", slug)
	return slug.strip("_")
