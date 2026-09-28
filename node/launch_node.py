"""Windows launcher: reuse the existing CUDA environment and stop on config errors."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def interpreter(config, folder):
    explicit = os.environ.get('PHYS_META_PYTHON') or config.get('python_executable')
    if explicit:
        path=Path(explicit)
        if not path.is_file():raise ValueError('Configured Python executable does not exist: '+str(path))
        return path
    src=Path(config.get('bubble_src') or Path(config.get('root',folder))/'bubble-ocr-app'/'src')
    for root in (src.parent, Path(config.get('root',folder)), folder):
        candidate=root/'.venv'/'Scripts'/'python.exe'
        if candidate.is_file():return candidate
    return Path(sys.executable)


def supervise(command, run=None, delay=None):
    run=run or subprocess.call;delay=delay or time.sleep
    # Exit 0 = user stop; 1 = config/auth error. Only unexpected crashes restart.
    for attempt in range(11):
        code=run(command)
        if code!=2 or attempt==10:return code
        print('Unexpected crash. Restarting in 5 seconds (%s/10)...'%(attempt+1),flush=True)
        delay(5)
    return 2


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=HERE/'inference-node.json')
    parser.add_argument('--check',action='store_true',help='Validate configuration and CUDA, without claiming a job')
    args,extra=parser.parse_known_args()
    try:
        cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
        python=interpreter(cfg,HERE)
        script=HERE/'agent.py'
        print('Node code:',script,flush=True);print('Python:',python,flush=True)
        print('Monitor: http://127.0.0.1:8901 - keep this window open; Ctrl+C to stop.',flush=True)
        if args.check:
            check=('import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);'
                   'import agent;agent.load_config(Path(sys.argv[2]));import torch;'
                   'print("Node version:",agent.AGENT_VERSION);print("CUDA available:",torch.cuda.is_available());'
                   'sys.exit(0 if torch.cuda.is_available() else 1)')
            return subprocess.call([str(python),'-c',check,str(HERE),str(args.config)])
        return supervise([str(python),str(script),'--config',str(args.config),*extra])
    except KeyboardInterrupt:return 0
    except (OSError,ValueError) as exc:
        print('Cannot start node:',exc,file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
