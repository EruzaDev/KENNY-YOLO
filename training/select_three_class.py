"""Select a three-class model using validation only and preserve legacy-class behavior."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from app.core.gestures import LEGACY_CLASSES

CLASSES = ('closed_fist', 'open_palm', 'thumbs_up')  # Historical selection protocol.


def preservation_failures(baseline: dict, candidate: dict) -> list[str]:
    if (baseline['split'] != 'val' or candidate['split'] != 'val'
            or baseline['manifest_sha256'] != candidate['manifest_sha256']
            or baseline['confidence'] != candidate['confidence']
            or set(baseline['conditions']) != set(candidate['conditions'])):
        return ['Validation protocols differ.']
    if tuple(candidate.get('classes', [])) != CLASSES or candidate.get('unsupported_classes'):
        return ['Candidate does not support the canonical three-class mapping.']
    failures = []
    for condition, previous in baseline['conditions'].items():
        current = candidate['conditions'][condition]
        for label in LEGACY_CLASSES:
            for metric in ['precision','recall']:
                if current['per_class'][label][metric] + 1e-12 < previous['per_class'][label][metric]:
                    failures.append(f'{condition}: {label} {metric} decreased')
    for view, previous in baseline['clean_views'].items():
        if candidate['clean_views'][view]['accuracy'] < previous['accuracy']:
            failures.append(f'Clean {view} accuracy decreased')
    if candidate['conditions']['clean']['per_class']['closed_fist']['recall'] < 1.:
        failures.append('Closed Fist clean validation recall is below 1.')
    if candidate['overall']['per_class']['closed_fist']['recall'] < .8:
        failures.append('Closed Fist stress recall is below .8.')
    if candidate['stability']['stable_correct_after_warmup'] < baseline['stability']['stable_correct_after_warmup']:
        failures.append('Synthetic sequence stabilization regressed')
    return failures



def fist_thumb_failures(baseline: dict, candidate: dict) -> list[str]:
    """Protect all existing classes and demand fewer actual fist/thumb confusions."""
    failures = preservation_failures(baseline, candidate)
    if failures == ['Validation protocols differ.']:
        return failures
    for condition, reference in baseline['conditions'].items():
        for metric in ['precision', 'recall']:
            if (candidate['conditions'][condition]['per_class']['closed_fist'][metric] + 1e-12
                    < reference['per_class']['closed_fist'][metric]):
                failures.append(f'{condition}: closed_fist {metric} decreased')
    def pair_errors(report):
        return sum(row['label'] in {'closed_fist', 'thumbs_up'}
                   and row['prediction'] in {'closed_fist', 'thumbs_up'}
                   and row['label'] != row['prediction'] for row in report['predictions'])
    if pair_errors(candidate) >= pair_errors(baseline):
        failures.append('No measured reduction in fist/thumb cross-confusions; collect real failure examples.')
    return failures


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=Path('outputs/three_class/baseline_validation_distance.json'))
    parser.add_argument('--candidates',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/three_class/selection.json'))
    parser.add_argument('--promote',action='store_true')
    parser.add_argument('--fist-thumb', action='store_true', help='Require all-class preservation and fewer fist/thumb cross-confusions.')
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Selection is already recorded; refusing to overwrite it.')
    baseline=json.loads(args.baseline.read_text());eligible=[];decisions=[]
    for path in args.candidates:
        report=json.loads(path.read_text())
        failures=(fist_thumb_failures if args.fist_thumb else preservation_failures)(baseline,report)
        decisions.append(dict(report=str(path),qualifies=not failures,reasons=failures))
        if not failures:eligible.append((path,report))
    if not eligible:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(selection_split='val', test_used=False, candidates=decisions,
                                              selected_report=None, promoted=False), indent=2))
        print(json.dumps(decisions,indent=2))
        raise SystemExit('No candidate meets the preservation checks. Live checkpoint was not changed.')
    path,report=min(eligible,key=lambda item:(-item[1]['overall']['per_class']['open_palm']['recall'],
                                            -item[1]['overall']['per_class']['closed_fist']['recall'],
                                            item[1]['overall']['group_nll']))
    checkpoint=Path(report['checkpoint']);metadata=json.loads(checkpoint.with_name('metadata.json').read_text())
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=report['checkpoint_sha256']:
        raise ValueError('Candidate checkpoint changed after evaluation.')
    result=dict(selection_split='val',test_used=False,candidates=decisions,selected_report=str(path),
                selected_sha256=report['checkpoint_sha256'],promoted=False,
                rule=('All three classes preserved and fewer actual fist/thumb cross-confusions required.' if args.fist_thumb else
                      'No per-condition legacy precision/recall regression; clean views and stability preserved; Closed Fist clean recall 1 and stress recall at least .8.'))
    if args.promote:
        if hashlib.sha256(Path('models/best.pt').read_bytes()).hexdigest()!=baseline['checkpoint_sha256']:
            raise ValueError('Live model changed during training; refusing to overwrite it.')
        clean=report['conditions']['clean']
        metadata.update(selection_report=str(args.output),robot_controls_enabled=False,
                        validation_summary=dict(image_count=clean['count'],independent_group_count=clean['group_count'],
                                                accuracy=clean['accuracy'],per_class=clean['per_class']),
                        final_test_status='Selection frozen; final reserved-group checks pending.')
        shutil.copy2(checkpoint,'models/best.pt.tmp')
        Path('models/metadata.json.tmp').write_text(json.dumps(metadata,indent=2))
        os.replace('models/best.pt.tmp','models/best.pt');os.replace('models/metadata.json.tmp','models/metadata.json')
        result['promoted']=True
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
