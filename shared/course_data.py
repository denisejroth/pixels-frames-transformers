"""Data helpers for Pixels, Frames & Transformers.

Module 02 builds the course corpora step by step. Later modules need the same images but not the
lesson, so they load this file and call one function per corpus:

    get_congress_portraits()   official portraits of current members of the US Congress (module 01)
    get_protest_corpus()       photos of demonstrations and protests in the Netherlands (module 02)
    get_packaging_corpus()     breakfast cereal packaging from Open Food Facts (module 02)
    get_koningsdag_corpus()    King's Day crowds, as non-protest examples (module 02, exercise)

It also contains `image_features()`, the handcrafted features from module 03, and the helpers
for the protest test set that runs through the course:

    protest_train_test()       your 40 coded images from module 04 as the test set, every other
                               image (with its category label) for training
    keep_my_files()            copy your codes and scoreboard to Google Drive (Colab)
    log_score(), show_scores() add a result to your scoreboard, and show it

Each function downloads what is missing into a local folder, skips files it already has, and
returns a manifest (a pandas DataFrame) with one row per image: where it came from, its license
and author, a checksum and the local path.

If the repository contains a frozen manifest for a corpus (shared/data/<name>_manifest.csv), the
functions download exactly the files listed there, so everyone works with the same images.
Otherwise they collect the corpus fresh from the APIs, the same way as module 02.
"""
import hashlib
import shutil
import io
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image

HEADERS = {"User-Agent": "PixelsFramesTransformers/1.0 (https://github.com/denisejroth/pixels-frames-transformers) course-notebook"}
RAW_BASE = "https://raw.githubusercontent.com/denisejroth/pixels-frames-transformers/main"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OFF_SEARCH = "https://world.openfoodfacts.org/api/v2/search"
PHOTO_URL = "https://raw.githubusercontent.com/unitedstates/images/gh-pages/congress/{size}/{bioguide}.jpg"
LEGISLATORS_URL = "https://raw.githubusercontent.com/unitedstates/congress-legislators/gh-pages/legislators-current.json"


# ── basics ────────────────────────────────────────────────────────────────────────────────
def download(url, params=None):
    """GET a URL (with optional query parameters); raise an error if it fails."""
    response = requests.get(url, params=params, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response


def safe_name(title, url):
    """A short, file-system-safe name based on the title, keeping the URL's file extension."""
    suffix = Path(urlparse(url).path).suffix.lower() or ".jpg"
    return hashlib.sha1(title.encode("utf-8")).hexdigest()[:12] + suffix


def download_images(table, folder, source, pause=0.5, verbose=True):
    """Download every `source_url` in `table` into `folder`; return the table as a manifest."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in enumerate(table.to_dict("records")):
        path = folder / safe_name(row["title"], row["source_url"])
        status = "ok"
        if not path.exists():
            try:
                path.write_bytes(download(row["source_url"]).content)
                time.sleep(pause)
            except requests.RequestException as error:
                status = f"failed: {error.__class__.__name__}"
        if status == "ok":
            data = path.read_bytes()
            try:
                with Image.open(io.BytesIO(data)) as img:
                    width, height = img.size
            except OSError:
                status = "failed: not an image"
            else:
                row.update(local_path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                           width=width, height=height)
        row.update(source=source, status=status,
                   downloaded_at=row.get("downloaded_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"))
        rows.append(row)
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1} of {len(table)} files")
    return pd.DataFrame(rows)


def _frozen_manifest(name):
    """Return the frozen manifest from the repository, or None if there isn't one."""
    try:
        return pd.read_csv(io.StringIO(download(f"{RAW_BASE}/shared/data/{name}_manifest.csv").text))
    except requests.RequestException:
        return None


# ── Wikimedia Commons ─────────────────────────────────────────────────────────────────────
def commons_api(params):
    return download(COMMONS_API, params={"format": "json", "formatversion": 2, **params}).json()


def category_members(category, member_type="file"):
    params = {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
              "cmtype": member_type, "cmlimit": 500}
    titles = []
    while True:
        data = commons_api(params)
        titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        if "continue" not in data:
            return titles
        params.update(data["continue"])
        time.sleep(0.5)


SKIP_WORDS = ["naked", "nude", "naakt", "wnbr"]


def collect_files(categories, depth=1, max_files=300):
    found, seen = {}, set()
    queue = [(c, 0) for c in categories]
    while queue and len(found) < max_files:
        category, level = queue.pop(0)
        if category in seen or any(w in category.lower() for w in SKIP_WORDS):
            continue
        seen.add(category)
        for title in category_members(category, "file"):
            if title.lower().endswith((".jpg", ".jpeg", ".png")):
                found.setdefault(title, category)
        if level < depth:
            for sub in category_members(category, "subcat"):
                queue.append((sub.removeprefix("Category:"), level + 1))
        time.sleep(0.5)
    return dict(list(found.items())[:max_files])


def _strip_html(text):
    return re.sub(r"<[^>]+>", "", text or "").strip()


def file_info(titles, found_in=None, thumb_width=500):
    records = []
    for start in range(0, len(titles), 50):
        data = commons_api({"action": "query", "titles": "|".join(titles[start:start + 50]),
                            "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
                            "iiurlwidth": thumb_width})
        for page in data["query"]["pages"]:
            if page.get("missing") or "imageinfo" not in page:
                continue
            info = page["imageinfo"][0]
            meta = info.get("extmetadata", {})
            value = lambda key: _strip_html(meta.get(key, {}).get("value", ""))
            records.append({
                "title": page["title"], "found_in": (found_in or {}).get(page["title"], ""),
                "source_url": info.get("thumburl", info["url"]), "page_url": info["descriptionurl"],
                "original_width": info["width"], "original_height": info["height"],
                "license": value("LicenseShortName"), "license_url": value("LicenseUrl"),
                "author": value("Artist"), "attribution_required": value("AttributionRequired") == "true",
                "date_taken": value("DateTimeOriginal"), "description": value("ImageDescription")[:200],
            })
        time.sleep(1)
    return pd.DataFrame(records)


def get_protest_corpus(folder="corpus/protests", max_files=300, years=range(2015, 2025)):
    """Photos from the Commons categories 'Demonstrations and protests in the Netherlands in <year>'."""
    frozen = _frozen_manifest("protests")
    if frozen is not None:
        print(f"Using the course's frozen list of {len(frozen)} protest photos.")
        table = frozen.drop(columns=[c for c in ["local_path", "status"] if c in frozen.columns])
    else:
        print("Collecting protest photos from Wikimedia Commons (this takes a few minutes)...")
        categories = [f"Demonstrations and protests in the Netherlands in {y}" for y in years]
        files = collect_files(categories, depth=1, max_files=max_files)
        table = file_info(list(files), found_in=files)
    manifest = download_images(table, folder, source="Wikimedia Commons")
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    print(f"{len(manifest)} protest photos ready in {folder}/")
    return manifest


def get_koningsdag_corpus(folder="corpus/koningsdag", max_files=60):
    """Photos of King's Day (Koningsdag) from a Commons search: crowds that are not protests."""
    frozen = _frozen_manifest("koningsdag")
    if frozen is not None:
        print(f"Using the course's frozen list of {len(frozen)} King's Day photos.")
        table = frozen.drop(columns=[c for c in ["local_path", "status"] if c in frozen.columns])
    else:
        print("Collecting King's Day photos from Wikimedia Commons...")
        data = commons_api({"action": "query", "list": "search", "srsearch": "Koningsdag filetype:bitmap",
                            "srnamespace": 6, "srlimit": max_files})
        titles = [hit["title"] for hit in data["query"]["search"]
                  if hit["title"].lower().endswith((".jpg", ".jpeg", ".png"))]
        table = file_info(titles, found_in={t: "search: Koningsdag" for t in titles})
    manifest = download_images(table, folder, source="Wikimedia Commons")
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    print(f"{len(manifest)} King's Day photos ready in {folder}/")
    return manifest


# ── Open Food Facts ───────────────────────────────────────────────────────────────────────
OFF_FIELDS = "code,product_name,brands,image_front_url,nutriscore_grade,nova_group,labels_tags"


def off_search(category, country="netherlands", pages=2, page_size=100):
    products = []
    for page in range(1, pages + 1):
        data = download(OFF_SEARCH, params={
            "categories_tags_en": category, "countries_tags_en": country, "fields": OFF_FIELDS,
            "sort_by": "unique_scans_n", "page_size": page_size, "page": page}).json()
        products += data["products"]
        if page * page_size >= data["count"]:
            break
        time.sleep(7)       # at most 10 search requests per minute
    return pd.DataFrame(products)


def get_packaging_corpus(folder="corpus/packaging", category="breakfast-cereals", pages=2):
    """Front-of-pack photos of breakfast cereals sold in the Netherlands, from Open Food Facts."""
    frozen = _frozen_manifest("packaging")
    if frozen is not None:
        print(f"Using the course's frozen list of {len(frozen)} products.")
        table = frozen.drop(columns=[c for c in ["local_path", "status"] if c in frozen.columns])
    else:
        print("Collecting packaging photos from Open Food Facts (this takes a few minutes)...")
        products = off_search(category, pages=pages)
        table = products.dropna(subset=["image_front_url"]).copy()
        table["code"] = table["code"].astype(str)
        table["title"] = "OFF " + table["code"]
        table["source_url"] = table["image_front_url"]
        table["page_url"] = "https://world.openfoodfacts.org/product/" + table["code"]
        table["license"] = "CC BY-SA"
        table["license_url"] = "https://world.openfoodfacts.org/terms-of-use"
        table["author"] = "Open Food Facts contributors"
        table["attribution_required"] = True
    manifest = download_images(table, folder, source="Open Food Facts", pause=1.0)
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    print(f"{len(manifest)} packaging photos ready in {folder}/")
    return manifest


# ── US Congress portraits ─────────────────────────────────────────────────────────────────
def get_congress_portraits(folder="portraits", n=200, size="225x275", seed=42):
    """Official portraits (public domain) of a random sample of current members of Congress."""
    rows = []
    for person in download(LEGISLATORS_URL).json():
        last = person["terms"][-1]
        rows.append({
            "bioguide": person["id"]["bioguide"],
            "name": person["name"].get("official_full", f"{person['name']['first']} {person['name']['last']}"),
            "chamber": {"sen": "Senate", "rep": "House"}[last["type"]],
            "party": last.get("party"), "state": last["state"], "gender": person["bio"].get("gender"),
            "first_year": int(person["terms"][0]["start"][:4]),
        })
    members = pd.DataFrame(rows)
    if n is not None and n < len(members):
        members = members.sample(n, random_state=seed).reset_index(drop=True)
    table = members.assign(title=members["bioguide"],
                           source_url=[PHOTO_URL.format(size=size, bioguide=b) for b in members["bioguide"]],
                           license="Public domain", author="US Government Publishing Office")
    manifest = download_images(table, folder, source="unitedstates/images", pause=0.05, verbose=False)
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    print(f"{len(manifest)} portraits ready in {folder}/")
    return manifest


# ── Handcrafted features (module 03) ──────────────────────────────────────────────────────
def load_rgb(path, max_side=500):
    """Open an image upright, in RGB, no larger than max_side pixels on its longest side."""
    from PIL import ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


_FACE_DETECTOR = None


def _face_detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        import cv2
        name = "haarcascade_frontalface_default.xml"
        path = Path(cv2.data.haarcascades) / name
        if not path.exists():
            path = Path(name)
            if not path.exists():
                path.write_bytes(download(f"https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/{name}").content)
        _FACE_DETECTOR = cv2.CascadeClassifier(str(path))
    return _FACE_DETECTOR


def image_features(path):
    """The light, color, composition and face features from module 03 (without OCR)."""
    import numpy as np
    import cv2
    img = load_rgb(path)
    arr = np.asarray(img, dtype=float)
    gray = np.asarray(img.convert("L"))
    gray_f = gray.astype(float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=75)
    min_side = max(12, int(min(gray.shape) * 0.03))
    faces = _face_detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
    areas = [w * h for (_, _, w, h) in faces]
    return {
        "brightness": gray_f.mean(),
        "contrast": gray_f.std(),
        "share_dark": (gray_f < 30).mean(),
        "share_bright": (gray_f > 225).mean(),
        "saturation": np.asarray(img.convert("HSV"), dtype=float)[:, :, 1].mean(),
        "colorfulness": np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2),
        "edge_density": (cv2.Canny(gray, 100, 200) > 0).mean(),
        "bytes_per_pixel": buffer.tell() / (img.width * img.height),
        "symmetry": 1 - np.abs(gray_f - gray_f[:, ::-1]).mean() / 255,
        "n_faces": len(faces),
        "largest_face_share": max(areas) / (img.width * img.height) if areas else 0.0,
    }


# ── Your codes and the shared test set (module 04 onward) ─────────────────────────────────
# Colab forgets files when a session ends. Your codes from module 04 are the test set for every
# later module, so keep_my_files() copies them (and your scoreboard) to Google Drive, and later
# modules find them there.
SEED = 42
MY_FILES = ("my_codes.csv", "my_recodes.csv", "my_results.csv")
DRIVE_DIR = Path("/content/drive/MyDrive/pixels-frames-transformers")


def _in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _mount_drive():
    """Mount Google Drive in Colab (asks for permission the first time). Returns True on success."""
    if not _in_colab():
        return False
    if Path("/content/drive/MyDrive").exists():
        return True
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        return Path("/content/drive/MyDrive").exists()
    except Exception as error:  # the user may decline
        print(f"Google Drive not mounted ({error.__class__.__name__}).")
        return False


def keep_my_files():
    """Copy my_codes.csv, my_recodes.csv and my_results.csv to Google Drive (in Colab)."""
    existing = [Path(name) for name in MY_FILES if Path(name).exists()]
    if not existing:
        print("Nothing to keep yet.")
        return
    if not _mount_drive():
        print("Keep these files next to the notebooks of later modules:", ", ".join(p.name for p in existing))
        return
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.copy(path, DRIVE_DIR / path.name)
    print(f"Saved {', '.join(p.name for p in existing)} to Google Drive (MyDrive/pixels-frames-transformers/).")


def restore_my_files():
    """Bring back my_codes.csv and friends from Google Drive if they are not here yet."""
    missing = [name for name in MY_FILES if not Path(name).exists()]
    if missing and _in_colab() and _mount_drive():
        for name in missing:
            if (DRIVE_DIR / name).exists():
                shutil.copy(DRIVE_DIR / name, name)
                print(f"Restored {name} from Google Drive.")


def coding_sample(protests, koningsdag, seed=SEED):
    """The 40 images coded in module 04: 30 protest photos and 10 King's Day photos, shuffled."""
    return pd.concat([
        protests.sample(30, random_state=seed).assign(category_label=1),
        koningsdag.sample(min(10, len(koningsdag)), random_state=seed).assign(category_label=0),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)


def protest_train_test(protests=None, koningsdag=None):
    """The protest task used from module 04 on.

    test:  your 40 coded images, with your `protest` code (or, if your codes can't be found, the
           category label as a stand-in)
    train: every other protest and King's Day image, with its category label
    """
    protests = protests if protests is not None else get_protest_corpus()
    koningsdag = koningsdag if koningsdag is not None else get_koningsdag_corpus()
    sample = coding_sample(protests, koningsdag)
    everything = pd.concat([protests.assign(category_label=1), koningsdag.assign(category_label=0)],
                           ignore_index=True)
    train = everything[~everything["title"].isin(sample["title"])].reset_index(drop=True)
    train = train[["title", "local_path", "category_label"]]

    restore_my_files()
    if Path("my_codes.csv").exists():
        codes = pd.read_csv("my_codes.csv")
        test = sample.merge(codes[["title", "protest"]], on="title")
        test.attrs["label_source"] = "your codes"
        if len(test) < len(sample):
            print(f"Note: {len(sample) - len(test)} coded images are no longer in the corpus; testing on {len(test)}.")
    else:
        print("WARNING: my_codes.csv not found, so the test set uses the category labels as a stand-in.\n"
              "Results then measure agreement with the Commons categories, not with what images show.\n"
              "Run module 04 (or upload my_codes.csv to this session) to test against your own codes.")
        test = sample.assign(protest=sample["category_label"])
        test.attrs["label_source"] = "category labels (stand-in)"
    test = test[["title", "local_path", "category_label", "protest"]].reset_index(drop=True)
    print(f"Training images: {len(train)} | test images: {len(test)} (labels: {test.attrs.get('label_source', '')})")
    return train, test


def log_score(method, y_true, y_pred, module=""):
    """Add a result on the 40 coded images to your scoreboard (my_results.csv)."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    row = {"module": module, "method": method,
           "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 3),
           "f1_protest": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 3),
           "n_test": len(y_true),
           "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = Path("my_results.csv")
    restore_my_files()
    table = pd.read_csv(path, dtype={"module": str}) if path.exists() else pd.DataFrame(columns=list(row))
    table = pd.concat([table[table["method"] != method], pd.DataFrame([row])], ignore_index=True)
    table.to_csv(path, index=False)
    return row


def show_scores():
    """Your scoreboard: every method you have scored on the 40 coded images."""
    restore_my_files()
    path = Path("my_results.csv")
    if not path.exists():
        print("No scores yet.")
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"module": str}).sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)
