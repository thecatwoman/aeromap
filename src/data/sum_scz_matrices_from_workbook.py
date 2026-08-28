import argparse
import csv
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS = {"main": MAIN_NS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read SCzF and SCzR matrices from an Excel workbook and save their summed map.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/nini/Downloads/Dallara aeromap copy.xlsx"),
        help="Path to the source workbook.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/tables/reference_table_sums"),
        help="Directory where CSV outputs will be written.",
    )
    return parser.parse_args()


def col_letters_to_number(col_letters: str) -> int:
    value = 0
    for char in col_letters:
        value = value * 26 + (ord(char) - 64)
    return value


def parse_cell_ref(cell_ref: str) -> tuple[int, int]:
    match = re.fullmatch(r"([A-Z]+)(\d+)", cell_ref)
    if not match:
        raise ValueError(f"Unexpected cell reference: {cell_ref}")
    col_letters, row_str = match.groups()
    return int(row_str), col_letters_to_number(col_letters)


def load_sheet_cells(workbook_path: Path) -> dict[tuple[int, int], str]:
    with zipfile.ZipFile(workbook_path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in shared_root.findall("main:si", NS):
                text_parts = [node.text or "" for node in si.iterfind(".//main:t", NS)]
                shared_strings.append("".join(text_parts))

        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        cells: dict[tuple[int, int], str] = {}
        for cell in sheet_root.findall(".//main:c", NS):
            ref = cell.attrib["r"]
            value_node = cell.find("main:v", NS)
            value = ""
            if value_node is not None and value_node.text is not None:
                value = value_node.text
                if cell.attrib.get("t") == "s":
                    value = shared_strings[int(value)]
            row, col = parse_cell_ref(ref)
            cells[(row, col)] = value
    return cells


def extract_matrix(cells: dict[tuple[int, int], str], marker_text: str) -> tuple[list[float], list[float], list[list[float]]]:
    marker_row = None
    for (row, _col), value in cells.items():
        if value == marker_text:
            marker_row = row
            break
    if marker_row is None:
        raise ValueError(f"Marker {marker_text!r} not found in workbook.")

    header_row = marker_row + 2
    data_start_row = marker_row + 3

    x_headers: list[float] = []
    col = 2
    while True:
        raw = cells.get((header_row, col), "")
        if raw == "":
            break
        x_headers.append(float(raw))
        col += 1

    if not x_headers:
        raise ValueError(f"No X headers found for {marker_text!r}.")

    y_headers: list[float] = []
    matrix: list[list[float]] = []
    row = data_start_row
    while True:
        raw_y = cells.get((row, 1), "")
        if raw_y == "":
            break
        y_headers.append(float(raw_y))
        values: list[float] = []
        for offset in range(len(x_headers)):
            raw = cells.get((row, 2 + offset), "")
            if raw == "":
                raise ValueError(f"Missing matrix value at row {row}, column {2 + offset} for {marker_text!r}.")
            values.append(float(raw))
        matrix.append(values)
        row += 1

    return x_headers, y_headers, matrix


def write_matrix_csv(path: Path, x_headers: list[float], y_headers: list[float], matrix: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + x_headers)
        for y_value, row_values in zip(y_headers, matrix, strict=True):
            writer.writerow([y_value] + row_values)


def main() -> None:
    args = parse_args()
    cells = load_sheet_cells(args.input)

    x_f, y_f, sczf = extract_matrix(cells, "(SCzF)")
    x_r, y_r, sczr = extract_matrix(cells, "(SCzR)")

    if x_f != x_r or y_f != y_r:
        raise ValueError("SCzF and SCzR matrices do not share the same headers, so they cannot be summed directly.")

    total = [
        [front + rear for front, rear in zip(front_row, rear_row, strict=True)]
        for front_row, rear_row in zip(sczf, sczr, strict=True)
    ]

    front_path = args.output_dir / "sczf_reference_table.csv"
    rear_path = args.output_dir / "sczr_reference_table.csv"
    total_path = args.output_dir / "scz_total_reference_table.csv"

    write_matrix_csv(front_path, x_f, y_f, sczf)
    write_matrix_csv(rear_path, x_r, y_r, sczr)
    write_matrix_csv(total_path, x_f, y_f, total)

    print(f"Saved SCzF table: {front_path}")
    print(f"Saved SCzR table: {rear_path}")
    print(f"Saved summed SCz table: {total_path}")


if __name__ == "__main__":
    main()
