"""Build a life list from a CSV of sightings.

Accepts an eBird "Download My Data" export (MyEBirdData.csv) or any CSV with
"Common Name" and "Scientific Name" columns. A "Taxonomic Order" or "Taxon
Order" column is optional and may be blank; duplicate rows are merged.

Follows eBird's life list rules: subspecies groups and forms roll up to their
species, while spuhs ("Buteo sp."), slashes ("Greater/Lesser Scaup"), hybrids
and domestic types are not counted.
"""
import csv
import re
from dataclasses import dataclass


@dataclass
class Species:
    common_name: str
    scientific_name: str
    taxonomic_order: float | None

    @property
    def slug(self):
        return re.sub(r"[^a-z0-9]+", "-", self.scientific_name.lower()).strip("-")


def to_species(common_name, scientific_name):
    """Return (common, binomial) for a countable taxon, or None if it doesn't count."""
    if "domestic" in common_name.lower() or " x " in scientific_name:
        return None
    tokens = scientific_name.split()
    if len(tokens) < 2:
        return None
    genus, epithet = tokens[0], tokens[1]
    if epithet == "sp." or "/" in genus or "/" in epithet or not epithet.isalpha():
        return None
    # "Northern Flicker (Yellow-shafted)" -> "Northern Flicker"
    common = re.sub(r"\s*\(.*\)$", "", common_name).strip()
    return common, f"{genus} {epithet}"


def load_life_list(csv_path):
    """Return the unique species in a sightings CSV, in taxonomic order.

    Species with no taxonomic order on any row sort last, alphabetically.
    """
    species = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            result = to_species(row["Common Name"], row["Scientific Name"])
            if result is None:
                continue
            common, binomial = result
            order_text = (row.get("Taxonomic Order") or row.get("Taxon Order") or "").strip()
            order = float(order_text) if order_text else None
            existing = species.get(binomial)
            if existing is None:
                species[binomial] = Species(common, binomial, order)
            # Take the lowest order seen, so the species row wins over its subspecies groups.
            elif order is not None and (existing.taxonomic_order is None or order < existing.taxonomic_order):
                existing.taxonomic_order = order
    return sorted(species.values(), key=lambda s: (s.taxonomic_order is None, s.taxonomic_order or 0, s.common_name))


if __name__ == "__main__":
    import sys

    life_list = load_life_list(sys.argv[1])
    for s in life_list:
        print(f"{s.common_name:30} {s.scientific_name}")
    print(f"{len(life_list)} species")
