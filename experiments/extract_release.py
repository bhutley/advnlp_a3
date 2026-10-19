"""Extract the split ACL19 release (ACL19_Release.z01 ... .z04 + .zip).

The archive was made with `zip -s`, which stores each member's offset relative
to the part it starts in. This concatenates the parts and corrects offsets by
each member's volume number, so no external zip tool is needed. Standard library only.
"""

import argparse
import shutil
import struct
import zipfile
from pathlib import Path


def mark_single_disk(path, last_disk):
    """zipfile refuses a ZIP64 locator that names several disks. Rewrite it to
    one disk; offsets are corrected per member afterwards."""
    with path.open("r+b") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 65558))
        tail = f.read()
        eocd = tail.rfind(b"PK\x05\x06")
        if eocd < 0:
            raise ValueError("End of central directory not found")
        cd_disk = struct.unpack("<H", tail[eocd + 6:eocd + 8])[0]
        locator = eocd - 20
        if tail[locator:locator + 4] != b"PK\x06\x07":
            return  # not ZIP64: zipfile does not check the disk count
        record = tail[locator - 56:locator]  # ZIP64 end record, no extensible data
        if record[:4] == b"PK\x06\x06":
            cd_disk = struct.unpack("<I", record[20:24])[0]
        if cd_disk != last_disk:
            raise ValueError("Central directory does not start on the last part")
        f.seek(size - len(tail) + locator + 4)
        f.write(struct.pack("<I", 0))
        f.seek(size - len(tail) + locator + 16)
        f.write(struct.pack("<I", 1))


def extract(archive_dir, out):
    archive_dir, out = Path(archive_dir), Path(out)
    last = archive_dir / "ACL19_Release.zip"
    parts = sorted(archive_dir.glob("ACL19_Release.z[0-9][0-9]")) + [last]
    if not last.is_file():
        raise ValueError(f"No ACL19_Release.zip in {archive_dir}")
    # Not *.zip, so a leftover can never be picked up as one of the parts.
    combined = archive_dir / "combined.tmp"
    starts, position = [], 0
    if not combined.exists() or combined.stat().st_size != sum(p.stat().st_size for p in parts):
        with combined.open("wb") as target:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, target, 16 << 20)
    for part in parts:
        starts.append(position)
        position += part.stat().st_size
    mark_single_disk(combined, len(parts) - 1)
    with zipfile.ZipFile(combined) as archive:
        # zipfile already shifted every offset by the last part's start.
        shift = starts[-1]
        todo = []
        for info in archive.infolist():
            info.header_offset += starts[info.volume] - shift
            # The overlap guard was computed from the uncorrected offsets.
            info._end_offset = None
            target = out / info.filename
            if info.is_dir() or not target.is_file() or target.stat().st_size != info.file_size:
                todo.append(info)
        for i, info in enumerate(todo, 1):
            archive.extract(info, out)
            if i % 5000 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
        print(f"Extracted {len(todo)} of {len(archive.infolist())} members into {out}")
    combined.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/archive")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()
    extract(args.archive, args.out)


if __name__ == "__main__":
    main()
