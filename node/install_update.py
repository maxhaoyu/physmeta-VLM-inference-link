"""Install a prepared release bundle into an existing node, keeping data and models."""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent

def active_node(folder):
    status=folder/'status.json'
    if not status.exists():return False
    pid=int(json.loads(status.read_text(encoding='utf-8'))['pid'])
    if os.name!='nt':return False
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        if ctypes.get_last_error()==5:raise ValueError('Cannot verify old node process; stop it before updating.')
        return False
    try:
        code=ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise ValueError('Cannot inspect old node process.')
        return code.value==259
    finally:kernel.CloseHandle(handle)

def install(target):
    target=target.resolve()
    if target==HERE.resolve():raise ValueError('Extract the update package into a separate folder, not the running node folder.')
    config=target/'inference-node.json'
    cfg=json.loads(config.read_text(encoding='utf-8-sig'))
    if active_node(target):raise ValueError('Old node is still running. Stop its console/scheduled task before updating.')
    source=Path(cfg.get('bubble_src') or Path(cfg.get('root',target))/'bubble-ocr-app'/'src')
    if not (source/'bubble_ocr/pipeline.py').is_file():raise ValueError('Existing bubble_src is not valid; configuration was not changed.')
    manifest=json.loads((HERE/'bundle-manifest.json').read_text())
    for rel,digest in manifest['files'].items():
        p=HERE/rel
        if p.resolve().is_relative_to(HERE.resolve()) is False:raise ValueError('Invalid bundle path')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Bundle verification failed: '+rel)
    pairs=[(HERE/n,target/n) for n in ['agent.py','launch_node.py','start-node.bat','vlm_fallback_windows.py']]
    pairs += [(p,source/p.relative_to(HERE/'bubble-src')) for p in (HERE/'bubble-src').rglob('*.py')]
    backup=target/'code-backups'/datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f');backup.mkdir(parents=True)
    originals=[]
    try:
        for i,(src,dest) in enumerate(pairs):
            old=backup/str(i)
            if dest.exists():shutil.copy2(dest,old)
            originals.append((dest,old))
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        (backup/'restore-map.json').write_text(json.dumps({str(d):str(b) if b.exists() else None for d,b in originals},indent=2),encoding='utf-8')
    except Exception:
        for dest,old in reversed(originals):
            if old.exists():shutil.copy2(old,dest)
            else:dest.unlink(missing_ok=True)
        raise
    print('Updated code:',target);print('Backup:',backup)
    print('Configuration, credentials, models, environments and task files were preserved.')
    print('Run start-node.bat --check first, then double-click start-node.bat.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('target',nargs='?',type=Path,default=Path(r'C:\PhysMetaInference-ocr'))
    try:install(parser.parse_args().target)
    except Exception as exc:print('Update stopped:',exc,file=sys.stderr);raise SystemExit(1)
