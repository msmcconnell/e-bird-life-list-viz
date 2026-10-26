"""Download research-grade, openly licensed iNaturalist photos for each species
on a life list.

Usage:
    python src/fetch_inat.py data/sample/MyEBirdData.csv --per-species 20

Output layout (re-running skips species that are already complete):
    <out>/<species-slug>/<photo_id>.jpg
    <out>/<species-slug>/meta.json   taxon match + per-photo license/attribution;
                                     set "manual": true to keep hand-added photos
    <out>/credits.csv                attribution for every downloaded photo
    <out>/unmatched.txt              species with no iNaturalist match or photos
"""
import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path

import requests

from life_list import load_life_list

API = "https://api.inaturalist.org/v1"
AVES_TAXON_ID = 3
# iNaturalist asks for at most ~1 request/second to the API.
API_INTERVAL = 1.0
DEFAULT_LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc"

session = requests.Session()
session.headers["User-Agent"] = "e-bird-life-list-viz (photo mosaic of a birder's life list)"
_last_api_call = 0.0


def api_get(path, **params):
    global _last_api_call
    for attempt in range(4):
        wait = API_INTERVAL - (time.monotonic() - _last_api_call)
        if wait > 0:
            time.sleep(wait)
        _last_api_call = time.monotonic()
        response = session.get(f"{API}{path}", params=params, timeout=30)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(5 * 2**attempt)
            continue
        response.raise_for_status()
        return response.json()["results"]
    response.raise_for_status()


def normalize(name):
    """Lowercase and strip accents, so "Choco Tinamou" matches "Chocó Tinamou"."""
    decomposed = unicodedata.normalize("NFKD", name or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()


def find_taxon(species):
    """Match a species to an active iNaturalist bird taxon.

    Tries the scientific name first (iNat resolves many synonyms), then the
    common name, since eBird and iNat taxonomies don't always agree. Only exact
    matches count: iNat's search is fuzzy and ranks by popularity, so its top
    hit for "Suiriri suiriri" is Eastern Kingbird (Spanish "Suirirí boreal").
    """
    filters = dict(rank="species", is_active="true", taxon_id=AVES_TAXON_ID, per_page=50)
    for query in (species.scientific_name, species.common_name):
        target = normalize(query)
        for taxon in api_get("/taxa", q=query, **filters):
            names = (taxon["name"], taxon.get("preferred_common_name"), taxon.get("matched_term"))
            if target in map(normalize, names):
                return taxon
    return None


def find_photos(taxon_id, count, licenses):
    """Return up to `count` photos, at most one per observation for variety."""
    observations = api_get(
        "/observations",
        taxon_id=taxon_id,
        quality_grade="research",
        photos="true",
        photo_license=licenses,
        order_by="votes",
        per_page=min(200, count * 2),
    )
    allowed = set(licenses.split(","))
    photos = []
    for obs in observations:
        photo = next((p for p in obs["photos"] if p.get("license_code") in allowed), None)
        if photo is None:
            continue
        photos.append({
            "photo_id": photo["id"],
            "url": photo["url"],
            "license": photo["license_code"],
            "attribution": photo["attribution"],
            "observer": (obs.get("user") or {}).get("login"),
            "observation_url": obs["uri"],
        })
        if len(photos) == count:
            break
    return photos


def download(url, path):
    response = session.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)


def fetch_species(species, out_dir, count, size, licenses):
    species_dir = out_dir / species.slug
    meta_path = species_dir / "meta.json"
    rejected = set()
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rejected = {p["photo_id"] for p in meta["photos"] if p.get("rejected")}
        # Hand-curated species (photos added by the user) are never refetched.
        if meta.get("manual"):
            return meta
        if len(meta["photos"]) >= count and all((species_dir / p["file"]).exists() for p in meta["photos"]):
            return meta

    taxon = find_taxon(species)
    if taxon is None:
        return None
    photos = find_photos(taxon["id"], count, licenses)

    species_dir.mkdir(parents=True, exist_ok=True)
    for photo in photos:
        photo["file"] = f"{photo['photo_id']}.jpg"
        photo["url"] = re.sub(r"/square\.", f"/{size}.", photo["url"])
        # Keep review decisions (see apply_review.py) across refetches.
        if photo["photo_id"] in rejected:
            photo["rejected"] = True
        target = species_dir / photo["file"]
        if not target.exists():
            download(photo["url"], target)

    meta = {
        "slug": species.slug,
        "common_name": species.common_name,
        "scientific_name": species.scientific_name,
        "taxonomic_order": species.taxonomic_order,
        "inat_taxon_id": taxon["id"],
        "inat_name": taxon["name"],
        "inat_common_name": taxon.get("preferred_common_name"),
        "photos": photos,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def write_credits(out_dir, metas):
    with open(out_dir / "credits.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "scientific_name", "file", "license", "attribution", "observation_url"])
        for meta in metas:
            for p in meta["photos"]:
                writer.writerow([meta["common_name"], meta["scientific_name"], f"{meta['slug']}/{p['file']}", p["license"], p["attribution"], p["observation_url"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ebird_csv", help="eBird 'Download My Data' export or any CSV with Common Name / Scientific Name columns")
    parser.add_argument("--out", default="data/images", help="output directory (default: data/images)")
    parser.add_argument("--per-species", type=int, default=20, help="photos per species (default: 20)")
    parser.add_argument("--size", default="medium", choices=["small", "medium", "large", "original"],
                        help="iNat photo size: small=240px, medium=500px, large=1024px (default: medium)")
    parser.add_argument("--licenses", default=DEFAULT_LICENSES,
                        help=f"comma-separated iNat license codes to accept (default: {DEFAULT_LICENSES})")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    life_list = load_life_list(args.ebird_csv)
    print(f"{len(life_list)} species on life list")

    metas, unmatched = [], []
    for i, species in enumerate(life_list, 1):
        meta = fetch_species(species, out_dir, args.per_species, args.size, args.licenses)
        if meta is None or not meta["photos"]:
            unmatched.append(species)
            status = "NO MATCH" if meta is None else "no photos"
        else:
            metas.append(meta)
            renamed = "" if meta["inat_name"] == species.scientific_name else f" (iNat: {meta['inat_name']})"
            status = f"{len(meta['photos'])} photos{renamed}"
        print(f"[{i}/{len(life_list)}] {species.common_name}: {status}")

    write_credits(out_dir, metas)
    (out_dir / "unmatched.txt").write_text(
        "".join(f"{s.common_name}\t{s.scientific_name}\n" for s in unmatched), encoding="utf-8")
    total = sum(len(m["photos"]) for m in metas)
    print(f"Done: {total} photos for {len(metas)} species; {len(unmatched)} unmatched")


if __name__ == "__main__":
    main()
