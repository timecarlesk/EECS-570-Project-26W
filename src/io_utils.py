"""CSV helpers."""

import csv
import os


def ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)


def write_csv(path, rows, fieldnames):
    ensure_parent(path)
    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)
