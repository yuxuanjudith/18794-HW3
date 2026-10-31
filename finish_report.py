"""Final processing for completed, full-validation HW3 experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_report import build
from vis_utils import VOC_CLASSES

ROOT=Path(__file__).resolve().parent


def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


def package_outputs(output):
    """Package reviewed artifacts without rerunning experiments or evaluation."""
    output=Path(output).resolve()
    prefix='HW3_Problems_1_4' if (ROOT/'report/sam_results.json').exists() else 'HW3_Problems_1_3'
    output.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'report/main.pdf',output/f'{prefix}.pdf')
    with zipfile.ZipFile(output/f'{prefix}_code_report.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob('*'):
            if not p.is_file(): continue
            if p in {output/f'{prefix}.pdf',output/f'{prefix}_code_report.zip'}: continue
            if output != ROOT and output.is_relative_to(ROOT) and p.is_relative_to(output): continue
            rel=p.relative_to(ROOT)
            if rel.parts[0] in {'checkpoints','results','output'} or '__pycache__' in rel.parts: continue
            if rel.parts[:2]==('datasets','data'): continue
            if p.suffix in {'.zip','.tmp','.aux','.out','.pyc'} or 'smoke' in p.name: continue
            if rel.parts[0]=='logs' and not p.name.startswith('deeplabv3plus_resnet50_'): continue
            z.write(p,str(rel))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir',type=Path,default=ROOT/'output',
                        help='Directory for the final PDF and code/report ZIP')
    output=parser.parse_args().output_dir.resolve()
    logs={}
    summary={}
    configs={}
    for loss in ['cross_entropy','focal_loss']:
        name=f'deeplabv3plus_resnet50_{loss}'
        configs[loss]=json.loads((ROOT/'logs'/f'{name}_config.json').read_text())
        assert configs[loss]['backbone_weights']=='v2' and configs[loss]['lr']==.02
        rows=json.loads((ROOT/'logs'/f'{name}_miou.json').read_text())
        assert rows[-1]['itrs']==5000
        assert all(row['validation_images']==1449 for row in rows)
        logs[loss]=rows
        best=max(rows,key=lambda row:row['mIoU'])
        ckpt=ROOT/'checkpoints'/f'best_{name}_VOC_os16.pth'
        run('evaluate_checkpoint.py','--ckpt',ckpt,'--exp_name',name,'--workers','2')
        measured=json.loads((ROOT/'logs'/f'{name}_evaluation.json').read_text())
        assert measured['validation_images']==1449
        if abs(measured['Mean IoU']-best['mIoU'])>.0001:
            raise RuntimeError('Independent checkpoint evaluation differs from saved validation result')
        summary[loss]=dict(best_miou=best['mIoU'],accuracy_at_best_miou=best['acc'],
            maximum_accuracy=max(r['acc'] for r in rows),best_epoch=best['epoch'],best_iteration=best['itrs'],
            total_iterations=rows[-1]['itrs'],validation_images=1449,
            independent_evaluation=measured,checkpoint=str(ckpt.relative_to(ROOT)),
            checkpoint_sha256=hashlib.file_digest(ckpt.open('rb'),'sha256').hexdigest(),
            train_seconds=sum(r['train_seconds'] for r in rows),
            validation_seconds=sum(r['val_seconds'] for r in rows))
    for key in ['model','output_stride','backbone_weights','lr','batch_size','crop_size',
                'total_itrs','random_seed','weight_decay','step_size','amp','channels_last',
                'backbone_weight_sha256','source_sha256']:
        if configs['cross_entropy'][key] != configs['focal_loss'][key]:
            raise RuntimeError(f'Loss comparison uses different {key}')
    summary['ce_threshold_passed']=summary['cross_entropy']['best_miou']>=.65
    (ROOT/'report/results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    figdir=ROOT/'report/figs'
    figdir.mkdir(exist_ok=True)
    styles={'cross_entropy':('#0072B2','o','Cross-entropy'),
            'focal_loss':('#D55E00','s','Class-weighted focal')}
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.facecolor':'white'}):
        for losses,stem in [(['cross_entropy'],'miou_ce'),(['focal_loss'],'miou_focal'),
                            (['cross_entropy','focal_loss'],'miou_both')]:
            fig,ax=plt.subplots(figsize=(6.4,3.8),layout='constrained')
            for loss in losses:
                rows=logs[loss]; color,marker,label=styles[loss]
                ax.plot([r['epoch'] for r in rows],[100*r['mIoU'] for r in rows],
                        color=color,marker=marker,markersize=3.5,label=label)
                b=max(rows,key=lambda row:row['mIoU'])
                ax.scatter(b['epoch'],100*b['mIoU'],s=60,facecolors='none',edgecolors=color,zorder=4)
            ax.axhline(65,color='#555555',ls='--',lw=1,label='Required CE mIoU: 65%')
            ax.set(xlabel='Epoch (last epoch is partial)',ylabel='Validation mIoU (%)',ylim=(0,100),
                   xlim=(.5,max(rows[-1]['epoch'] for rows in logs.values())+.5))
            ax.grid(alpha=.2); ax.legend(loc='lower right',fontsize=9)
            for fmt in ['png','pdf']:
                fig.savefig(figdir/f'{stem}.{fmt}',dpi=300)
            plt.close(fig)
    run('visualize.py','samples')
    run('visualize.py','compare','--ckpt1',ROOT/summary['cross_entropy']['checkpoint'],
        '--ckpt2',ROOT/summary['focal_loss']['checkpoint'])
    class_rows=[]
    for i,name in enumerate(VOC_CLASSES):
        class_rows.append({'class':name,
            'cross_entropy_iou':summary['cross_entropy']['independent_evaluation']['Class IoU'][str(i)],
            'focal_iou':summary['focal_loss']['independent_evaluation']['Class IoU'][str(i)]})
    with (ROOT/'report/per_class_iou.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(class_rows[0]));writer.writeheader();writer.writerows(class_rows)
    provenance={'dataset':'VOC2012 official segmentation train=1464, val=1449',
        'statistics':'training masks; void label 255 excluded',
        'curves':'raw per-epoch full-validation logs, no smoothing; final epoch partial',
        'comparison_images':'first five preselected foreground classes in report/sample_ids.json',
        'evaluation':'original resolution; single-scale FP32; includes background; no flips',
        'replicates':'one independently initialized run per loss, seed 1',
        'source_logs':[f'logs/deeplabv3plus_resnet50_{loss}_miou.json' for loss in logs]}
    (ROOT/'report/figure_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    build(draft=False)
    tectonic=ROOT.parents[1]/'tools/tectonic/tectonic.exe'
    if not tectonic.exists():
        executable=shutil.which('tectonic')
        if not executable:
            raise RuntimeError('LaTeX source is ready. Install Tectonic or compile report/main.tex in Overleaf.')
        tectonic=Path(executable)
    with (ROOT/'report/compile_console.log').open('w',encoding='utf-8') as f:
        subprocess.run([str(tectonic),'--keep-logs',str(ROOT/'report/main.tex')],check=True,stdout=f,stderr=subprocess.STDOUT)
    package_outputs(output)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
