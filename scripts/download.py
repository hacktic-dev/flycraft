import hashlib, json, urllib.request, zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
def fetch(url, path, digest):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        print('Downloading',path.name,flush=True)
        tmp=path.with_suffix('.partial')
        urllib.request.urlretrieve(url,tmp)
        tmp.replace(path)
    with path.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual != digest: raise RuntimeError(f'Checksum mismatch: {path}')
    print('Verified',path.name,flush=True)
def archive(url,name,digest):
    p=ROOT/'.tools'/name
    fetch(url,p,digest)
    with zipfile.ZipFile(p) as z:
        top=ROOT/'.tools'/z.namelist()[0].split('/')[0]
        executable=top/'bin'/('java.exe' if name=='jdk.zip' else 'clang++.exe')
        if not executable.exists(): z.extractall(ROOT/'.tools')
    print('Extracted',name,flush=True)
def data():
    source=ROOT/'vendor/doomfly/data-provenance/malecns_v1/source.lock.json'
    lock=json.loads(source.read_text())
    out=ROOT/'vendor/doomfly/connectome_data/malecns_v1'
    for name,meta in lock.items(): fetch(meta['url'],out/name,meta['sha256'])
    (out/'source.lock.json').write_text(source.read_text())
if __name__=='__main__':
    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(data),pool.submit(archive,'https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12.1%2B1/OpenJDK21U-jdk_x64_windows_hotspot_21.0.12.1_1.zip','jdk.zip','f9d6e191ab098c0d416e7d588a24420a8621cd2f4720dab2459b8b7b2d2d8b4e'),pool.submit(archive,'https://github.com/mstorsjo/llvm-mingw/releases/download/20260908/llvm-mingw-20260908-ucrt-x86_64.zip','llvm.zip','1bcf74d06b724aeecaa6412ca85f5b26fb1da770e7cdcefa9263c9c5c3ad34b6')]
        for j in jobs: j.result()

