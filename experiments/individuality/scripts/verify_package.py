from pathlib import Path
import hashlib,csv,sys
root=Path(__file__).resolve().parents[1]
manifest=root/'PACKAGE_FILE_MANIFEST.csv'
if not manifest.exists():
    print('manifest not found'); sys.exit(1)
bad=[]
for r in csv.DictReader(manifest.open(encoding='utf-8-sig')):
    p=root/r['relative_path']
    if not p.exists(): bad.append((r['relative_path'],'missing')); continue
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    if h!=r['sha256']: bad.append((r['relative_path'],'sha'))
print({'checked':sum(1 for _ in csv.DictReader(manifest.open(encoding='utf-8-sig'))),'bad':bad})
sys.exit(1 if bad else 0)
