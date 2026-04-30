from html import escape


def render_mlflow_note(content: str) -> str:
	lines = content.splitlines()
	rendered: list[str] = []
	index = 0
	while index < len(lines):
		if _starts_markdown_table(lines, index):
			table_html, index = _render_markdown_table(lines, index)
			rendered.append(table_html)
			continue
		rendered.append(lines[index])
		index += 1
	return "\n".join(rendered).strip()


def _starts_markdown_table(lines: list[str], index: int) -> bool:
	if index + 1 >= len(lines):
		return False
	return _is_table_row(lines[index]) and _is_separator_row(_parse_table_cells(lines[index + 1]))


def _render_markdown_table(lines: list[str], index: int) -> tuple[str, int]:
	headers = _parse_table_cells(lines[index])
	rows: list[list[str]] = []
	cursor = index + 2
	while cursor < len(lines) and _is_table_row(lines[cursor]):
		rows.append(_parse_table_cells(lines[cursor]))
		cursor += 1

	header_html = "".join(f"<th>{_render_cell(cell)}</th>" for cell in headers)
	body_html = "".join(
		"<tr>" + "".join(f"<td>{_render_cell(cell)}</td>" for cell in row) + "</tr>"
		for row in rows
	)
	table_html = (
		"<table>"
		"<thead><tr>"
		+ header_html
		+ "</tr></thead>"
		+ "<tbody>"
		+ body_html
		+ "</tbody></table>"
	)
	return table_html, cursor


def _parse_table_cells(line: str) -> list[str]:
	return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_table_row(line: str) -> bool:
	stripped = line.strip()
	return stripped.startswith("|") and stripped.endswith("|")


def _is_separator_row(cells: list[str]) -> bool:
	return all(cell and set(cell.replace(":", "")) == {"-"} for cell in cells)


def _render_cell(value: str) -> str:
	rendered = escape(value, quote=False)
	for raw in ("&lt;br&gt;", "&lt;br/&gt;", "&lt;br /&gt;"):
		rendered = rendered.replace(raw, "<br/>")
	return rendered
