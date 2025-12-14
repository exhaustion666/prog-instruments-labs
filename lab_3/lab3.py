import re
import csv
from checksum import calculate_checksum, serialize_result
from typing import List, Dict


VALIDATION_PATTERNS = {
    "telephone": r"^\+7-\([0-9]{3}\)-[0-9]{3}-[0-9]{2}-[0-9]{2}$",
    "height": r"^[0-2]\.[0-9]{2}$",
    "snils": r"^\d{11}$",
    "identifier": r"^\d{2}-\d{2}/\d{2}$",
    "occupation": r"^[А-Яа-яA-Za-zЁё\s\-.,()/&+]+$",
    "longitude": r"^-?(?:180(?:\.0+)?|1[0-7][0-9](?:\.[0-9]+)?|[1-9]?[0-9](?:\.[0-9]+)?)$",
    "blood_type": r"^(?:A|B|AB|O)[+-]$",
    "issn": r"^\d{4}-\d{4}$",
    "locale_code": r"^[a-z]{2}(?:-[a-z]{2})?$",
    "date": r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])$",
}


def clean_value(value: str) -> str:
    """
    Cleans the value from extra characters.
    
    :param value: Raw value from CSV file
    :return: Cleaned value without quotes, BOM, and null bytes
    """
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value.replace('\ufeff', '').replace('\x00', '')


def validate_row(row: Dict[str, str], headers: List[str]) -> bool:
    """
    Validates a single row of data.
    
    :param row: Dictionary with row data
    :param headers: List of column headers
    :return: True if all fields in the row are valid, otherwise False
    """
    for header in headers:
        if header not in VALIDATION_PATTERNS:
            continue

        pattern = VALIDATION_PATTERNS[header]
        value = row.get(header, '')

        if not re.match(pattern, value):
            return False

    return True


def validate_csv_file(file_path: str, variant: int) -> None:
    """
    Main function for CSV file validation.
    
    :param file_path: Path to the CSV file
    :param variant: Variant number
    :return: None
    """
    invalid_rows = []

    with open(file_path, 'r', encoding='utf-16') as file:
        reader = csv.reader(file, delimiter=';')
        headers_raw = next(reader)
        headers = [clean_value(h) for h in headers_raw]

        for row_num, row in enumerate(reader):
            cleaned_row = {}
            for i, header in enumerate(headers):
                if i < len(row):
                    cleaned_row[header] = clean_value(row[i])
                else:
                    cleaned_row[header] = ''

            if not validate_row(cleaned_row, headers):
                invalid_rows.append(row_num)
                
    checksum = calculate_checksum(invalid_rows)
    serialize_result(variant, checksum)


if __name__ == "__main__":
    validate_csv_file("52.csv", 52)
