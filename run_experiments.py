"""Run both required experiments sequentially, then generate figures and report.

Reruns retain completed experiments and resume others at the last saved epoch.
"""
import ctypes
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def status(**kwargs):
    payload = dict(updated=datetime.datetime.now().astimezone().isoformat(), pid=os.getpid(), **kwargs)
    tmp = ROOT / 'logs/pipeline_status.tmp'
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    tmp.replace(ROOT / 'logs/pipeline_status.json')


def main():
    os.chdir(ROOT)
    (ROOT/'logs').mkdir(exist_ok=True)
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for loss in ['cross_entropy', 'focal_loss']:
            name = f'deeplabv3plus_resnet50_{loss}'
            log = ROOT/'logs'/f'{name}_miou.json'
            history = json.loads(log.read_text()) if log.exists() else []
            ckpt = ROOT/'checkpoints'/f'latest_{name}_VOC_os16.pth'
            best_ckpt = ROOT/'checkpoints'/f'best_{name}_VOC_os16.pth'
            if history and history[-1]['itrs'] >= 5000 and best_ckpt.exists():
                continue
            if history and history[-1]['itrs'] >= 5000 and ckpt.exists() and not best_ckpt.exists():
                raise RuntimeError(f'{name}: completed run is missing its best checkpoint. Restore it, '
                                   'or back up its checkpoints and logs before starting a fresh run.')
            # The submission ZIP includes measured logs but deliberately excludes large weights.
            # Preserve those reference results when a fresh extraction is trained again.
            if not ckpt.exists():
                previous = [p for p in (ROOT/'logs').glob(f'{name}_*') if p.is_file()]
                if previous:
                    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    archive = ROOT/'logs/previous_runs'/f'{name}_{stamp}'
                    archive.mkdir(parents=True)
                    for record in previous:
                        shutil.copy2(record,archive/record.name)
                    print(f'No checkpoint for {name}; preserved existing logs in {archive}. '
                          'Starting from ImageNet initialization.',flush=True)
            command = [sys.executable, '-u', 'main.py', '--amp', '--channels_last',
                       '--workers', '2', '--backbone_weights', 'v2', '--lr', '0.02',
                       '--loss_type', loss, '--exp_name', name]
            if loss == 'focal_loss':
                command += ['--class_weights', 'class_weights.npy']
            if ckpt.exists():
                command += ['--continue_training', '--ckpt', str(ckpt)]
            with open(ROOT/'logs'/f'{name}_console.log', 'a' if ckpt.exists() else 'w', encoding='utf-8') as stream:
                stream.write('\nCOMMAND: '+subprocess.list2cmdline(command)+'\n')
                stream.flush()
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
                status(state='training', loss=loss, child_pid=process.pid, command=command)
                code = process.wait()
            if code:
                raise RuntimeError(f'{loss} exited with code {code}; see its console log')
        status(state='postprocessing')
        subprocess.run([sys.executable, 'finish_report.py'], check=True)
        status(state='complete')
    except Exception as exc:
        status(state='failed', error=str(exc))
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
