"""Fill the supplied course LaTeX template using measured results only."""
import argparse
import json
import re
import zipfile
from pathlib import Path
from vis_utils import VOC_CLASSES

ROOT = Path(__file__).resolve().parent
REPORT = ROOT/'report'


def box(s):
    return '\\begin{tcolorbox}[breakable,colback=white,colframe=black,boxrule=0.4pt]\n'+s+'\n\\end{tcolorbox}\n'


def figure(name, caption, width='\\linewidth'):
    return r'\begin{center}\includegraphics[width='+width+']{figs/'+name+'}\n'+r'\end{center}{\small '+caption+'}\n'


def build(draft=False):
    stats = json.loads((REPORT/'class_stats.json').read_text())
    records = {}
    for loss in ['cross_entropy','focal_loss']:
        path = ROOT/'logs'/f'deeplabv3plus_resnet50_{loss}_miou.json'
        rows = json.loads(path.read_text()) if path.exists() else []
        if not draft and (not rows or rows[-1]['itrs'] != 5000 or any(x['validation_images'] != 1449 for x in rows)):
            raise RuntimeError(f'{loss}: final report requires 5000 steps and full validation at every epoch')
        records[loss] = rows
    best = {k:max(v,key=lambda x:x['mIoU']) for k,v in records.items() if v}
    bg = 100*stats['background']['pixel_share']
    foreground = {k:v for k,v in stats.items() if k != 'background'}
    most = max(foreground,key=lambda k:foreground[k]['pixel_share'])
    least = min(foreground,key=lambda k:foreground[k]['pixel_share'])
    ratio=foreground[most]['pixel_share']/foreground[least]['pixel_share']
    p11 = r'''The implementation loads each JPEG as an RGB PIL image and keeps the PNG segmentation
mask as a single-channel array of class indices (including the palette-indexed PNG format).
It applies the same geometric transform to image and mask, and returns the pair.
The RGB image is normalized with ImageNet mean $(0.485,0.456,0.406)$ and standard deviation
$(0.229,0.224,0.225)$. Images use bilinear resizing and masks use nearest-neighbor resizing.
Label 255 denotes void pixels and is excluded from the loss and metrics. Padded mask pixels
also receive 255. The color decoder indexes the VOC palette, yielding an $H\times W\times3$ RGB visualization.
Code: \texttt{datasets/voc.py} and \texttt{utils/ext\_transforms.py}.'''
    p12 = (rf'''There are \textbf{{20 foreground object categories plus one background class}}, so the
network predicts 21 labels. The official segmentation split contains 1,464 training images
and 1,449 validation images. All statistics below use only training masks and exclude void pixels.

The dataset is \textbf{{not class-balanced}}. Background covers \textbf{{{bg:.2f}\%}} of labeled pixels.
The most frequent foreground class, {most}, accounts for {100*foreground[most]['pixel_share']:.2f}\%,
whereas {least} accounts for {100*foreground[least]['pixel_share']:.2f}\%, a {ratio:.1f}-fold difference.
'''+r'''
Frequent, large regions contribute many more pixel losses than rare or small objects. A model
can obtain high pixel accuracy by favoring background while missing these objects. Small
objects can also disappear after downsampling, and boundaries are sensitive to localization
errors. Class-aware loss weighting and emphasizing difficult pixels are therefore relevant.

\begin{center}\small\input{class_stats.tex}\end{center}
{\small Pixel percentages use all non-void training pixels as the denominator. Image counts
mean images containing at least one non-void pixel of that class, not object-instance counts.}''')
    p13 = r'''Consider an image with 90 background pixels and 10 foreground pixels. A model that
predicts background everywhere obtains $90/100=90\%$ accuracy, despite missing the entire object.
Its background IoU is $90/100=0.90$, foreground IoU is $0/10=0$, and two-class mIoU is
$(0.90+0)/2=0.45$. The lower mIoU reveals the failed foreground prediction.

For class $c$, $\mathrm{IoU}_c=TP_c/(TP_c+FP_c+FN_c)$.
Pixel accuracy weights classes by their pixel counts; mIoU averages the class IoUs with equal
class weight. For this assignment, the confusion matrix is accumulated across the full
validation set, void pixels are excluded, and mIoU includes background among the 21 classes.'''
    p14 = r'''The panel contains one training example for each of the 20 foreground classes and one
additional background example. Each adjacent pair is RGB image followed by ground-truth mask;
each complete row contains ten images. Examples were chosen before model evaluation by
maximizing the target class's image-area fraction, without reusing an image.
The final row contains the remaining background pair. Colors use the VOC palette; the light
boundary color denotes void pixels, which are ignored in quantitative evaluation.
''' + figure('samples_per_class.png',r'Training examples and unmodified ground-truth labels. Image IDs are saved in \texttt{report/sample\_ids.json}.')
    p21 = r'''ASPP combines five parallel branches at output stride 16: a $1\times1$ convolution,
three $3\times3$ atrous convolutions with dilation rates 6, 12, and 18, and a global-average-pooling
branch. Every branch produces 256 channels with batch normalization and ReLU. The pooled
feature is bilinearly resized back to the feature-map size. Concatenating the five outputs
produces 1,280 channels; a $1\times1$ projection, batch normalization, ReLU, and dropout 0.1
reduce this to 256 channels. Atrous branches capture different context scales and the pooling
branch contributes image-level context \cite{deeplabv3}.

Both complete networks passed a forward/backward check with a $2\times3\times65\times73$
input and returned $2\times21\times65\times73$ logits, verifying odd-size alignment and gradient flow.'''
    p22 = r'''\textbf{DeepLabV3 head.} The final ResNet feature (2,048 channels) enters ASPP, followed
by a $3\times3$ convolution, batch normalization, ReLU, and a $1\times1$ classifier. The resulting
21-channel logits are bilinearly upsampled to the original image size.

\textbf{DeepLabV3+ head.} In addition to the ASPP output, the decoder uses the low-level
\texttt{layer1} feature at output stride 4. A $1\times1$ convolution reduces its 256 channels to
48. The ASPP feature is resized to that spatial resolution and concatenated with the low-level
feature, producing 304 channels. A $3\times3$ convolution with batch normalization and ReLU,
then a $1\times1$ classifier, produces the logits before final upsampling.

The principal difference is the decoder's explicit fusion of low-level spatial detail with
high-level semantic context. This gives V3+ a route to refine object boundaries \cite{deeplabv3plus}.
The supplied course implementation uses a ResNet-50 backbone and standard convolutions.
Code: \texttt{network/deeplab.py}, \texttt{network/modeling.py}, and \texttt{network/utils.py}.'''
    p31 = r'''The optimizer is SGD with momentum 0.9 and weight decay $10^{-4}$. Two parameter
groups assign learning rate $2\times10^{-3}$ to the ImageNet-pretrained backbone and $2\times10^{-2}$ to the
new segmentation head. The backbone therefore updates at one tenth of the head learning rate.'''
    p32 = r'''\texttt{StepLR(step\_size=1000, gamma=0.1)} is advanced after each successful optimizer
update. The head learning rates for updates 1--1000, 1001--2000, 2001--3000, 3001--4000, and
4001--5000 are respectively $2\times10^{-2},2\times10^{-3},2\times10^{-4},2\times10^{-5},2\times10^{-6}$; the backbone learning
rates remain one tenth of these. A mixed-precision overflow skips both the optimizer update
and the scheduler step, so 5,000 counted iterations mean 5,000 actual optimizer updates.'''
    setup = r'''Both experiments independently initialize the same DeepLabV3+ ResNet-50 from
torchvision ImageNet-1K V2 classification backbone weights
(\texttt{resnet50-11ad3fa6.pth}) \cite{resnetweights}; the segmentation head starts randomly. These weights use
no segmentation annotations. Settings are shared:
seed 1, batch size 8, crop $513\times513$, output stride 16, random scaling in $[0.5,2.0]$,
random horizontal flip with probability 0.5, and the SGD/StepLR settings above.
Training uses FP16 autocast with gradient scaling on an RTX 5060 Ti 8GB. Validation uses
FP32, batch size 1, the original image resolution, all 1,449 validation images, and no
multi-scale inference or test-time flips. The entire validation confusion matrix determines
each reported mIoU. With 1,464 training images, each full epoch has 183 batches. An AMP overflow
can skip a batch update; such skips are logged and do not count toward the 5,000 updates.
Validation occurs after every full data pass and once after the final partial pass.
Model selection uses validation mIoU.

An earlier exploratory run used ImageNet V1 and head learning rate 0.01. Its best full-validation
mIoU was 61.59\% through 11 completed epochs (2,012 updates). It was stopped before 5,000 updates
and is not one of the two completed experiments compared here. The final configuration uses
ImageNet V2 and head learning rate 0.02; the original split, optimizer ratio and step schedule
are preserved. The exploratory log is retained in \texttt{report/pilot\_v1\_miou.json}.'''
    def result_text(loss):
        rows=records[loss]
        if not rows or rows[-1]['itrs'] < 5000:
            return r'\textbf{Draft: this experiment has not finished; no final performance is claimed.}'
        b=best[loss]
        return (rf"Best validation mIoU is \textbf{{{100*b['mIoU']:.2f}\%}} at epoch {b['epoch']} "
                rf"(update {b['itrs']}); pixel accuracy at the same checkpoint is {100*b['acc']:.2f}\%. "
                rf"The run completed exactly {rows[-1]['itrs']:,} optimizer updates.")
    p33=setup+'\n\n'+result_text('cross_entropy')
    if records['cross_entropy'] and records['cross_entropy'][-1]['itrs']==5000:
        meets=best['cross_entropy']['mIoU']>=.65
        p33 += '\n\n'+(r'The best mIoU satisfies the required 65\% threshold.' if meets else
                         r'The measured best mIoU is below the required 65\% threshold; this run is reported as measured.')
    if (REPORT/'figs/miou_ce.png').exists():
        p33 += figure('miou_ce.png', 'Cross-entropy validation mIoU after each epoch. The last point is the final partial epoch.')
    p34=r'''The alternative loss is \textbf{class-weighted focal loss}:
\[
L=-\frac{1}{|V|}\sum_{i\in V}\alpha_{y_i}(1-p_{i,y_i})^\gamma\log p_{i,y_i},\qquad \gamma=2,
\]
where $V$ contains non-void pixels. For raw training-set class frequency $f_c$,
$\tilde\alpha_c=1/\sqrt{\max(f_c,10^{-8})}$ and
$\alpha_c=\tilde\alpha_c/\sum_k f_k\tilde\alpha_k$.
Thus the frequency-weighted mean class weight is one. The square root tempers the weight
assigned to rare classes; the focal factor suppresses already-correct, easy pixels \cite{focal}.
This extends cross-entropy beyond the built-in loss and addresses the imbalance measured in
Problem 1. The class weights are computed once from training labels only. For example,
background weight is 0.382 and bicycle weight is 5.916. The implementation handles ignored
pixels and computes log-softmax in FP32 under autocast. Setting $\gamma=0$ and unit class
weights numerically matches cross-entropy, including its gradients.

The second model is trained independently under the same settings as the first, rather than
fine-tuning the cross-entropy model.
'''+'\n\n'+result_text('focal_loss')
    if len(best)==2 and all(v and v[-1]['itrs']==5000 for v in records.values()):
        delta=100*(best['focal_loss']['mIoU']-best['cross_entropy']['mIoU'])
        if delta>=0:
            p34+=rf'''\par Focal loss improves best validation mIoU by {delta:.2f} percentage points in
these runs. This is consistent with greater emphasis on hard or underrepresented pixels.
The comparison changes both focal modulation and class weighting, so it does not separately
identify the contribution of either component. It is a single-seed comparison.'''
        else:
            p34+=rf'''\par Focal loss reduces best validation mIoU by {-delta:.2f} percentage points in
these runs. Balancing frequencies does not guarantee better overlap: increased emphasis on
rare or hard pixels may increase false positives, and down-weighting easy pixels changes the
effective gradient scale. The same rapidly decaying learning-rate schedule may therefore be
less suitable for the alternative objective. These are plausible explanations, not causal
conclusions established by this single-seed comparison.'''
        p34+=r'''\par The table below compares the dominant background, most frequent foreground,
and least frequent foreground classes, selected using training-set pixel frequencies from
Problem 1. IoUs use each model's best-mIoU checkpoint. All 21 class IoUs are included in
\texttt{report/per\_class\_iou.csv}.
\begin{center}\small\begin{tabular}{lrrr}\hline
Class & Train pixels (\%) & CE IoU (\%) & Focal IoU (\%)\\\hline
'''
        for name in ['background',most,least]:
            index=str(VOC_CLASSES.index(name))
            ce_iou=100*best['cross_entropy']['class_iou'][index]
            focal_iou=100*best['focal_loss']['class_iou'][index]
            p34+=f"{name} & {100*stats[name]['pixel_share']:.2f} & {ce_iou:.2f} & {focal_iou:.2f}"+r'\\'+'\n'
        p34+=r'\hline\end{tabular}\end{center}'
        improved_classes=sum(best['focal_loss']['class_iou'][str(i)]>
                             best['cross_entropy']['class_iou'][str(i)] for i in range(21))
        p34+=rf'\par Focal loss achieves a higher IoU for {improved_classes} of the 21 classes in this comparison. '
        least_index=str(VOC_CLASSES.index(least))
        if best['focal_loss']['class_iou'][least_index]<best['cross_entropy']['class_iou'][least_index]:
            p34+=rf'''The least frequent foreground class, {least}, also has lower IoU despite its
larger loss weight. Thus this weighting scheme does not deliver the intended rare-class benefit
for that class under the fixed training settings.'''
        result_file=REPORT/'results.json'
        if result_file.exists():
            measurements=json.loads(result_file.read_text())
            ce=measurements['cross_entropy']['independent_evaluation']
            fo=measurements['focal_loss']['independent_evaluation']
            if 'background_as_foreground_rate' in ce and 'background_as_foreground_rate' in fo:
                p34+=(rf"\par Reloading both best checkpoints gives background-to-foreground error rates "
                      rf"of {100*ce['background_as_foreground_rate']:.2f}\% (CE) and "
                      rf"{100*fo['background_as_foreground_rate']:.2f}\% (focal), measured over true "
                      rf"background pixels. Conversely, {100*ce['foreground_as_background_rate']:.2f}\% "
                      rf"and {100*fo['foreground_as_background_rate']:.2f}\% of true foreground pixels "
                      rf"are predicted as background. These rates come from the saved full-validation "
                      rf"confusion matrices and exclude void pixels.")
                if fo['background_as_foreground_rate']>ce['background_as_foreground_rate']:
                    p34+=r''' The higher background false-alarm rate is consistent with the lower
background weight and stronger emphasis on foreground pixels; it demonstrates one cost of
this alternative loss under the shared training schedule.'''
                    if fo['foreground_as_background_rate']<ce['foreground_as_background_rate']:
                        p34+=r''' Fewer foreground pixels are missed, while more background pixels
receive object labels. This foreground-recall versus false-alarm tradeoff explains why stronger
emphasis on foreground pixels does not automatically improve overlap quality.'''
    if (REPORT/'figs/miou_focal.png').exists():
        p34+=figure('miou_focal.png','Class-weighted focal-loss validation mIoU, evaluated with the same protocol as cross-entropy.')
    p35=r'''Performance is measured on the held-out validation split. Accuracy labeled
``Acc. there'' is measured at the checkpoint selected by best mIoU. Maximum accuracy over all
evaluated checkpoints is also provided to distinguish the two selection rules.'''
    if len(best)==2 and all(v and v[-1]['itrs']==5000 for v in records.values()):
        table=[r'\begin{center}\small\begin{tabular}{lrrrr}\hline',
               r'Loss & Best mIoU (\%) & Acc. there (\%) & Max acc. (\%) & Epoch\\\hline']
        for key,label in [('cross_entropy','Cross-entropy'),('focal_loss','Weighted focal')]:
            b=best[key]
            table.append(f"{label} & {100*b['mIoU']:.2f} & {100*b['acc']:.2f} & {100*max(r['acc'] for r in records[key]):.2f} & {b['epoch']}"+r'\\')
        table+=[r'\hline\end{tabular}\end{center}']
        p35+='\n'+'\n'.join(table)
    else:
        p35+=r'\par\textbf{Draft: final comparison awaits both complete experiments.}'
    qualitative=''
    if (REPORT/'figs/compare.png').exists():
        p35+=figure('compare.png','The same five training examples selected in Problem 1: aeroplane, bicycle, bird, boat, and bottle. These illustrate predictions; numerical performance above uses the separate validation set.')
        qualitative=r'''\par Both models identify the dominant object class in these examples. The CE model
keeps more of the background around the aeroplane and bicycle, while focal predictions spread
the object labels into larger background regions. The bird prediction shows a similar tendency.
Both models lose fine gaps and some boundary detail, including background near the bottom of
the bottle image. These training-image observations illustrate the error patterns; the quantitative
comparison and error rates are measured on the held-out validation split.'''
    if (REPORT/'figs/miou_both.png').exists():
        p35+=figure('miou_both.png',r'Both unsmoothed validation trajectories on a shared scale. The dashed horizontal line denotes 65\% mIoU.')
    p35+=qualitative
    sam_path=REPORT/'sam_results.json'
    sam_done=sam_path.exists() and (REPORT/'figs/compare_sam.png').exists()
    p4=r'Problem 4 is outside the requested Problems 1--3 scope and has not been completed.'
    if sam_done:
        sam=json.loads(sam_path.read_text())
        assert len(sam['images'])==5
        assert sam['model']=='vit_b' and sam['points_per_batch']==16
        counts=', '.join(str(x['masks']) for x in sam['images'])
        p4=r'''The original Segment Anything Model (SAM), with the official pretrained ViT-B
checkpoint, is run without fine-tuning on the same five training images used in Problem 3.5
\cite{sam}. The reference to Problem 2.5 in the question is interpreted as Problem 3.5.
Automatic mask generation uses a $32\times32$ grid of point prompts, 16 prompts per batch,
predicted-IoU threshold 0.88, stability threshold 0.95, and no extra crop layers. Inference is
FP32 on the same RTX 5060 Ti. No ground-truth mask or ground-truth box is supplied as a prompt.
''' + rf' The retained mask counts, in row order, are {counts}. ' + r'''
For display, each mask receives a deterministic arbitrary color. Larger masks are drawn first
and smaller masks last; overlapping masks can therefore obscure one another. Black means no
retained mask covers that pixel. Colors do not represent VOC categories and are not consistent
semantic labels across rows. The original binary masks and scores are saved separately.

\textbf{Semantic segmentation versus SAM.} Our DeepLabV3+ models learn a fixed set of 21 VOC
labels and assign a category to every pixel. SAM predicts binary regions conditioned on prompts;
automatic generation samples many prompts and filters the resulting masks. Its regions can
represent whole objects, parts, or background surfaces, can overlap, and do not identify labels
such as ``person'' or ``bicycle.'' SAM therefore does not directly replace a VOC semantic classifier.
The plotted SAM colors cannot be scored as VOC class IDs. Computing a comparable 21-class mIoU
would require an additional, explicitly defined semantic-label assignment procedure.
'''
        p4+=figure('compare_sam.png','Same five images: RGB, ground truth, cross-entropy, weighted focal, and SAM. SAM colors indicate individual masks, not semantic classes.')
        p4+=r'''\par In these examples, SAM retains small regions such as aeroplane windows and
lettering, bicycle components, and bottle-label details. It also separates surfaces such as
the sail and surrounding water. A single VOC object may be split into many SAM masks.
Large black areas, particularly on the bird and aeroplane, show that this automatic configuration
does not retain a complete mask for every visible region. Point or box prompts targeting a
particular object, or different filtering settings, can produce different masks. The comparison
illustrates the difference between category prediction and class-agnostic region extraction;
it does not establish that either method is universally more accurate.'''
    answers=[p12,p13,p14,p22,p33,p34,p35,p4]
    tex=(REPORT/'template_original.tex.txt').read_text(encoding='utf-8')
    matches=re.findall(r'\\begin\{comment\}.*?\\end\{comment\}',tex,flags=re.S)
    assert len(matches)==len(answers),(len(matches),len(answers))
    it=iter(answers)
    tex=re.sub(r'\\begin\{comment\}.*?\\end\{comment\}',lambda _:box(next(it)),tex,flags=re.S)
    tex=tex.replace(r'\usepackage{graphicx}',r'\usepackage{graphicx}'+'\n'+r'\usepackage[margin=0.8in]{geometry}',1)
    tex=tex.replace(r'\begin{document}',r'\hypersetup{colorlinks=true,urlcolor=blue,citecolor=blue}'+'\n'+r'\begin{document}',1)
    tex=tex.replace('hhttps://www.cmu.edu','https://www.cmu.edu')
    tex=tex.replace(r'\section{',r'\clearpage'+'\n'+r'\section{')
    tex=tex.replace('2. (5pts) How many categories',box(p11)+'\n2. (5pts) How many categories',1)
    tex=tex.replace('2. (15pts) Complete DeepLabHead',box(p21)+'\n2. (15pts) Complete DeepLabHead',1)
    tex=tex.replace('2. (5pts) Build the learning rate scheduler.',box(p31)+'\n2. (5pts) Build the learning rate scheduler.',1)
    tex=tex.replace('3. (15pts) Train the DeepLabV3+',box(p32)+'\n3. (15pts) Train the DeepLabV3+',1)
    tex=tex.replace('3. (5pts) Semantic Segmentation',r'\clearpage'+'\n3. (5pts) Semantic Segmentation',1)
    tex=tex.replace('4. (10pts) Based on what',r'\clearpage'+'\n4. (10pts) Based on what',1)
    tex=tex.replace('5. (5pts) For the trained models',r'\clearpage'+'\n5. (5pts) For the trained models',1)
    tex=re.sub(r'\\begin\{figure\}\[!htb\].*?\\end\{figure\}', '', tex, flags=re.S)
    scope=r'\par\medskip\noindent\textbf{Scope: Problems 1--3.} Student name: \underline{\hspace{4cm}}\par'
    if sam_done:
        scope=scope.replace('1--3','1--4')
    if draft:
        scope+=r'\noindent\textbf{Working draft: training is in progress.}\par'
    tex=tex.replace(r'\section{Prepare Data Pipeline',scope+'\n'+r'\section{Prepare Data Pipeline',1)
    references=r'''
\clearpage
\begin{thebibliography}{9}
\bibitem{deeplabv3} L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam.
Rethinking Atrous Convolution for Semantic Image Segmentation. arXiv:1706.05587, 2017.
\bibitem{deeplabv3plus} L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam.
Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ECCV, 2018.
\bibitem{focal} T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Doll\'ar.
Focal Loss for Dense Object Detection. ICCV, 2017.
\bibitem{voc} M. Everingham et al. The PASCAL Visual Object Classes Challenge 2012.
\url{https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/}.
\bibitem{resnetweights} PyTorch. Torchvision ResNet-50 model and ImageNet-1K weights.
\url{https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html}.
\bibitem{skills} T. Kassis, V. Agarwal, Y. He, D. Patel, and A. M. Brueckner.
Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, 2026.
\url{https://doi.org/10.48550/arXiv.2609.00065}.
\bibitem{sam} A. Kirillov et al. Segment Anything. ICCV, 2023.
\url{https://github.com/facebookresearch/segment-anything}.
\end{thebibliography}
\noindent\textbf{Software and assistance.} The course starter package and supplied implementations
were the starting point. OpenAI Codex assisted with code review, implementation fixes, local
experiment execution, figure generation, and preparation of this report. The Scientific
Visualization skill informed figure review and provenance preservation. Raw logs, configuration,
image-selection IDs, and numerical checks accompany the report. The student should review
all code and explanations and complete their identifying information before submission.
'''
    tex=tex.replace(r'\end{document}',references+'\n'+r'\end{document}')
    (REPORT/'main.tex').write_text(tex,encoding='utf-8')
    print('Wrote',REPORT/'main.tex','draft=',draft)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--draft',action='store_true')
    build(p.parse_args().draft)
