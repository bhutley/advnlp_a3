"""Build calls.csv, segments.csv and call_texts.csv from the ACL19 release.

The release has one folder per call, named <Company Name>_<YYYYMMDD>, holding
TextSequence.txt (one executive sentence per line, in time order) and CEO/ with
one <Speaker>_<paragraph>_<sentence>.mp3 per line. Audio files sort numerically by
(paragraph, sentence); a plain string sort would put _10_1 before _2_1.

Calls whose sentence and audio counts differ are excluded and reported.
Requires pandas only.
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd

TEXT_NAMES = ("TextSequence.txt", "Text.txt")
AUDIO_DIRS = ("CEO", "Audio")
AUDIO_NAME = re.compile(r"_(\d+)_(\d+)\.(mp3|wav|flac)$", re.IGNORECASE)


def audio_order(paths):
    keyed = []
    for path in paths:
        match = AUDIO_NAME.search(path.name)
        if not match:
            raise ValueError(f"Unexpected audio file name: {path}")
        keyed.append(((int(match.group(1)), int(match.group(2))), path))
    keys = [k for k, _ in keyed]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate paragraph/sentence numbers in {paths[0].parent}")
    return [path for _, path in sorted(keyed)]


def parse_call_id(folder_name):
    name, _, date = folder_name.rpartition("_")
    if not name or not re.fullmatch(r"\d{8}", date):
        raise ValueError(f"Folder is not <Company>_<YYYYMMDD>: {folder_name}")
    return name, pd.Timestamp(date).date().isoformat()


def build(release, out):
    release, out = Path(release), Path(out)
    folders = sorted(p for p in release.iterdir() if p.is_dir())
    if not folders:
        raise ValueError(f"No call folders under {release}")
    calls, segments, texts, problems = [], [], [], []
    for folder in folders:
        call_id = folder.name
        name, call_date = parse_call_id(call_id)
        # The full release uses TextSequence.txt and CEO/; the GitHub sample
        # uses Text.txt and Audio/.
        text_file = next((folder / n for n in TEXT_NAMES if (folder / n).is_file()), None)
        audio_dir = next((folder / n for n in AUDIO_DIRS if (folder / n).is_dir()), None)
        audio = [p for p in audio_dir.iterdir() if p.is_file()] if audio_dir else []
        if text_file is None or not audio:
            problems.append((call_id, "missing transcript or audio folder"))
            continue
        lines = [line.strip() for line in text_file.read_text(encoding="utf-8", errors="replace").splitlines()]
        lines = [line for line in lines if line]
        audio = audio_order(audio)
        if len(lines) != len(audio):
            problems.append((call_id, f"{len(lines)} sentences but {len(audio)} audio files"))
            continue
        segment_dir = out / "segments" / call_id
        segment_dir.mkdir(parents=True, exist_ok=True)
        for i, (line, audio_path) in enumerate(zip(lines, audio)):
            text_path = segment_dir / f"{i:04d}.txt"
            text_path.write_text(line, encoding="utf-8")
            segments.append({
                "call_id": call_id,
                "text_path": Path(os.path.relpath(text_path, out)).as_posix(),
                "audio_path": Path(os.path.relpath(audio_path, out)).as_posix(),
            })
        calls.append({"call_id": call_id, "name": name, "call_date": call_date})
        texts.append({"call_id": call_id, "text": " ".join(lines)})
    return pd.DataFrame(calls), pd.DataFrame(segments), pd.DataFrame(texts), problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--companies", default="acl19_proc/data/company.csv")
    parser.add_argument("--smoke", type=int, default=10,
                        help="Also write segments_smoke.csv for the first N priced calls")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    calls, segments, texts, problems = build(args.release, out)
    calls.to_csv(out / "calls.csv", index=False)
    segments.to_csv(out / "segments.csv", index=False)
    texts.to_csv(out / "call_texts.csv", index=False)
    pd.DataFrame(problems, columns=["call_id", "reason"]).to_csv(out / "manifest_excluded.csv", index=False)
    companies = pd.read_csv(args.companies)
    unknown = sorted(set(calls.name) - set(companies.name))
    priced = calls[calls.name.isin(companies.name[companies.status == "ok"])]
    smoke_ids = set(priced.call_id.head(args.smoke))
    segments[segments.call_id.isin(smoke_ids)].to_csv(out / "segments_smoke.csv", index=False)
    print(f"{len(calls)} calls from {calls.name.nunique()} companies, {len(segments)} segments; "
          f"excluded {len(problems)} calls")
    print(f"{len(priced)} calls have price data; {len(unknown)} company names not in company.csv")
    for name in unknown:
        print(f"  not in company.csv: {name}")


if __name__ == "__main__":
    main()
