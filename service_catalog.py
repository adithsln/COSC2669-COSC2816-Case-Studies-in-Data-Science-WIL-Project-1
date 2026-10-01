"""Load and validate the official support-service catalogue."""

import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

BASE_DIR = Path(__file__).resolve().parent
CATALOG_FILE = BASE_DIR / "data" / "official_services.csv"

REQUIRED_COLUMNS = [
    "id",
    "institution",
    "title",
    "needs",
    "summary",
    "eligibility",
    "contact",
    "url",
]

ALLOWED_INSTITUTIONS = {
    "RMIT_SHARED",
    "RMIT_UNIVERSITY",
    "RMIT_UP",
    "MONASH",
    "MELBOURNE",
    "GENERAL",
}

ALLOWED_HOSTS = {
    "www.rmit.edu.au",
    "www.monash.edu",
    "services.unimelb.edu.au",
    "www.esafety.gov.au",
    "www.lifeline.org.au",
}

def split_needs(value):
    """Convert the pipe-separated needs field into a list."""
    return [item.strip() for item in value.split("|") if item.strip()]

def load_catalog(path=CATALOG_FILE):
    """Load the service catalogue and check that every record is usable."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Support-service catalogue not found: {path}"
        )

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "official_services.csv has no header row."
            )

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in reader.fieldnames
        ]

        if missing_columns:
            raise ValueError(
                "official_services.csv is missing required columns: "
                + ", ".join(missing_columns)
            )

        records = []
        seen_ids = set()

        for row_number, row in enumerate(reader, start=2):
            record = {
                column: (row.get(column) or "").strip()
                for column in REQUIRED_COLUMNS
            }

            empty_fields = [
                column
                for column, value in record.items()
                if not value
            ]

            if empty_fields:
                raise ValueError(
                    f"Row {row_number} has empty required fields: "
                    + ", ".join(empty_fields)
                )

            service_id = record["id"]

            if service_id in seen_ids:
                raise ValueError(
                    "Duplicate service ID in official_services.csv: "
                    f"{service_id}"
                )

            seen_ids.add(service_id)

            institution = record["institution"]

            if institution not in ALLOWED_INSTITUTIONS:
                raise ValueError(
                    f"Unexpected institution for {service_id}: "
                    f"{institution}"
                )

            source_url = urlsplit(record["url"])

            if (
                source_url.scheme != "https"
                or source_url.netloc.lower() not in ALLOWED_HOSTS
            ):
                raise ValueError(
                    f"Unsupported official source URL for {service_id}: "
                    f"{record['url']}"
                )

            record["needs"] = split_needs(record["needs"])

            if not record["needs"]:
                raise ValueError(
                    f"No support-topic tags were supplied for {service_id}."
                )

            records.append(record)

    if not records:
        raise ValueError(
            "official_services.csv is empty."
        )

    return records

def snapshot_hash(records):
    """Create a stable hash so the vector store can detect catalogue changes."""
    catalogue_text = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        catalogue_text.encode("utf-8")
    ).hexdigest()

def as_context(record):
    """Convert one service record into the text used for retrieval."""
    return "\n".join(
        [
            f"Service ID: {record['id']}",
            f"Service: {record['title']}",
            f"Institution: {record['institution']}",
            f"Relevant needs: {', '.join(record['needs'])}",
            f"Official-source summary: {record['summary']}",
            f"Eligibility: {record['eligibility']}",
            f"Contact: {record['contact']}",
            f"Official page: {record['url']}",
        ]
    )
