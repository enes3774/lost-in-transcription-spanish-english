"""Parallel zip, step 2: stitch part archives (each written by Info-ZIP) into ONE archive without recompressing.
Every entry's local header + compressed data is copied byte for byte; only the central directory is rebuilt (with the new
offsets). mode 'bins': split the package's files into N lists for N zip processes (the one huge Voxtral file gets its own).
usage: merge_zips.py bins <package_dir> <work_dir> <n>   |   merge_zips.py merge <out.zip> <part.zip> [...]  (parts are deleted as consumed)"""
import os
import sys
import zipfile

if sys.argv[1] == "bins":
    root, work, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
    files = []
    for d, _, fs in os.walk(root, followlinks=True):
        for f in fs:
            p = os.path.join(d, f)
            files.append((os.path.getsize(p), os.path.relpath(p, root)))
    files.sort(reverse=True)
    bins, load = [[] for _ in range(n)], [0.0] * n
    bins[0].append(files[0][1]); load[0] = float("inf")                      # the biggest file alone
    for size, rel in files[1:]:
        cost = size * (1.6 if "/omni/" in rel else 1.0)                      # Omni's 4-bit weights deflate slowest
        i = min(range(1, n), key=lambda k: load[k])
        bins[i].append(rel); load[i] += cost
    for i, b in enumerate(bins):
        open(os.path.join(work, f"bin_{i}.lst"), "w").write("\n".join(b) + "\n")
        print(f"bin {i}: {len(b)} files, {sum(s for s, r in files if r in set(b)) / 1e9:.1f} GB")
    print(f"{len(files)} files in total")
    sys.exit(0)

out, parts = sys.argv[2], sys.argv[3:]
assert not os.path.exists(out), out
dst = zipfile.ZipFile(out, "w", allowZip64=True)
seen = set()
for part in parts:
    with zipfile.ZipFile(part) as src:
        infos = sorted(src.infolist(), key=lambda z: z.header_offset)
        ends = [z.header_offset for z in infos[1:]] + [src.start_dir]
        with open(part, "rb") as f:
            for z, end in zip(infos, ends):
                if z.filename in seen:
                    continue
                seen.add(z.filename)
                f.seek(z.header_offset)
                new_off = dst.fp.tell()
                left = end - z.header_offset
                assert left >= 30 + len(z.filename.encode()) + z.compress_size, z.filename
                while left:
                    buf = f.read(min(left, 64 << 20))
                    assert buf, "short read"
                    dst.fp.write(buf); left -= len(buf)
                z.header_offset = new_off
                dst.filelist.append(z); dst.NameToInfo[z.filename] = z
    os.remove(part)
    print(f"{os.path.basename(part)}: copied, {len(seen)} entries so far, {dst.fp.tell() / 1e9:.2f} GB", flush=True)
dst.start_dir = dst.fp.tell()
dst._didModify = True
dst.close()
with zipfile.ZipFile(out) as z:
    names = z.namelist()
    assert len(names) == len(set(names))
    print(f"MERGED_OK {len(names)} entries, {os.path.getsize(out)} bytes, methods {sorted(set(i.compress_type for i in z.infolist()))}")
