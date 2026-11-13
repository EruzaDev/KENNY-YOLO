"""Compare one candidate on inherited validation groups only, then optionally promote."""
import argparse
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image
from app.services.detector import Detector
from training.metrics import classification_metrics


def score(model_path: Path, records: list[dict], confidence: float) -> dict:
    detector = Detector(model_path)
    detector.load()
    if not records or any(record['split'] != 'val' for record in records):
        raise ValueError('Only nonempty validation records may enter model selection.')
    with Image.open(records[0]['processed']) as image:
        detector.predict(image.convert('RGB'), confidence)
    rows = []
    for record in records:
        if record['split'] != 'val':
            raise ValueError('Only validation records may enter model selection.')
        with Image.open(record['processed']) as image:
            output = detector.predict(image.convert('RGB'), confidence)
        scores = output['scores']
        if not scores:
            raise ValueError('A labeled validation image was rejected as empty; review the data.')
        prediction = output['predictions'][0]['class_name'] if output['predictions'] else 'unknown'
        rows.append(dict(path=record['processed'], label=record['label'], view=record['view'],
                         group=record['group'], prediction=prediction, scores=scores,
                         latency_ms=output['latency_ms'],
                         nll=-math.log(max(scores[record['label']], 1e-12))))
    groups = defaultdict(list)
    for row in rows:
        groups[row['group']].append(row)
    _, report = classification_metrics([r['label'] for r in rows],
                                       [r['prediction'] for r in rows], detector.classes + ['unknown'])
    views = {}
    for view in sorted({r['view'] for r in rows}):
        subset = [r for r in rows if r['view'] == view]
        views[view] = dict(count=len(subset), accuracy=sum(r['prediction']==r['label'] for r in subset)/len(subset),
                           nll=sum(r['nll'] for r in subset)/len(subset))
    return dict(checkpoint_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                image_count=len(rows), independent_group_count=len(groups), confidence=confidence,
                accuracy=report['accuracy'], per_class={c:report[c] for c in detector.classes},
                mean_latency_ms=sum(r['latency_ms'] for r in rows)/len(rows),
                group_nll=sum(sum(r['nll'] for r in group)/len(group) for group in groups.values())/len(groups),
                all_views_correct_group_rate=sum(all(r['prediction']==r['label'] for r in group)
                                                 for group in groups.values())/len(groups),
                views=views, predictions=rows)


def is_improvement(baseline: dict, candidate: dict) -> bool:
    return (candidate['accuracy'] >= baseline['accuracy']
            and candidate['all_views_correct_group_rate'] >= baseline['all_views_correct_group_rate']
            and all(candidate['views'][v]['accuracy'] >= baseline['views'][v]['accuracy'] for v in baseline['views'])
            and all(candidate['per_class'][c][key] >= baseline['per_class'][c][key]
                    for c in baseline['per_class'] for key in ['precision','recall'])
            and candidate['group_nll'] < baseline['group_nll'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=Path('models/baseline/best.pt'))
    parser.add_argument('--candidate',type=Path,default=Path('models/candidate_originals/best.pt'))
    parser.add_argument('--manifest',type=Path,default=Path('data/processed_with_originals/manifest.json'))
    parser.add_argument('--output',type=Path,default=Path('outputs/improvement/validation_comparison.json'))
    parser.add_argument('--confidence',type=float,default=.8)
    parser.add_argument('--promote',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit('Comparison already recorded; refusing to overwrite model selection evidence.')
    manifest=json.loads(args.manifest.read_text())
    baseline_metadata=json.loads(args.baseline.with_name('metadata.json').read_text())
    candidate_metadata=json.loads(args.candidate.with_name('metadata.json').read_text())
    if (manifest['source_manifest_sha256'] != baseline_metadata['dataset_manifest_sha256']
            or hashlib.sha256(args.manifest.read_bytes()).hexdigest() != candidate_metadata['dataset_manifest_sha256']):
        raise ValueError('Candidate/baseline dataset lineage does not match this validation manifest.')
    validation=[r for r in manifest['manifest'] if r['split']=='val']
    baseline=score(args.baseline,validation,args.confidence)
    candidate=score(args.candidate,validation,args.confidence)
    better=is_improvement(baseline,candidate)
    result=dict(split='val',test_used=False,baseline=baseline,candidate=candidate,
                selection_rule='No reduction in class precision/recall, view accuracy or group correctness; lower group NLL.',
                candidate_qualifies=better,promoted=bool(better and args.promote),
                limitations=[f"Only {candidate['independent_group_count']} validation groups; this is weak evidence of real-world accuracy.",
                             'No measured negatives or new-camera test set.',
                             'Positive-only confidence improvement does not establish calibration on non-target inputs.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    if better and args.promote:
        metadata=dict(candidate_metadata, selection_report=str(args.output),
                      validation_summary={k:v for k,v in candidate.items() if k!='predictions'},
                      final_test_status='Not evaluated on a new independent test set after this improvement.')
        destination=Path('models')
        shutil.copy2(args.candidate,destination/'best.pt.tmp')
        (destination/'metadata.json.tmp').write_text(json.dumps(metadata,indent=2))
        os.replace(destination/'best.pt.tmp',destination/'best.pt')
        os.replace(destination/'metadata.json.tmp',destination/'metadata.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ['baseline','candidate']},indent=2))
    for name,data in [('baseline',baseline),('candidate',candidate)]:
        print(name,json.dumps({k:v for k,v in data.items() if k not in ['predictions','per_class']},indent=2))


if __name__=='__main__':
    main()
