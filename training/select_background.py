"""Promote a stress-tested candidate without regressing either gesture or existing views."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from training.robustness import qualifies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', type=Path, default=Path('outputs/background/baseline.json'))
    parser.add_argument('--candidate-reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/background/selection.json'))
    parser.add_argument('--promote', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Selection is already recorded; refusing to overwrite it.')
    baseline = json.loads(args.baseline_report.read_text())
    candidates, decisions = [], []
    for path in args.candidate_reports:
        candidate = json.loads(path.read_text())
        checkpoint = Path(candidate['checkpoint'])
        metadata = json.loads(checkpoint.with_name('metadata.json').read_text())
        if (hashlib.sha256(checkpoint.read_bytes()).hexdigest() != candidate['checkpoint_sha256']
                or metadata['initial_checkpoint_sha256'] != baseline['checkpoint_sha256']):
            raise ValueError('Candidate checkpoint or initial weight provenance differs from its report.')
        valid, failures = qualifies(baseline, candidate)
        decisions.append(dict(report=str(path), qualifies=valid, reasons=failures))
        if valid:
            candidates.append((path, candidate, metadata))
    selected = min(candidates, key=lambda item: (
        -item[1]['overall']['per_class']['open_palm']['recall'], item[1]['overall']['group_nll'])) if candidates else None
    result = dict(selection_split='val', test_used=False, confidence=baseline['confidence'],
                  baseline_sha256=baseline['checkpoint_sha256'], candidates=decisions,
                  selected_report=str(selected[0]) if selected else str(args.baseline_report),
                  selected_sha256=selected[1]['checkpoint_sha256'] if selected else baseline['checkpoint_sha256'],
                  promoted=False,
                  rule='No per-condition precision/recall regression in either class; preserve clean views and sequence stability; improve Open Palm background recall.')
    if selected and args.promote:
        if hashlib.sha256(Path('models/best.pt').read_bytes()).hexdigest() != baseline['checkpoint_sha256']:
            raise ValueError('Live weights changed during training; refusing to overwrite them.')
        path, report, metadata = selected
        clean = report['conditions']['clean']
        metadata.update(selection_report=str(args.output), robustness_validation_report=str(path),
                        validation_summary=dict(image_count=clean['count'], independent_group_count=clean['group_count'],
                                                accuracy=clean['accuracy'], per_class=clean['per_class'],
                                                confidence=report['confidence'], group_nll=clean['group_nll']),
                        final_test_status='Selection frozen; reserved-group final checks pending.')
        shutil.copy2(Path(report['checkpoint']), 'models/best.pt.tmp')
        Path('models/metadata.json.tmp').write_text(json.dumps(metadata, indent=2))
        os.replace('models/best.pt.tmp', 'models/best.pt')
        os.replace('models/metadata.json.tmp', 'models/metadata.json')
        result['promoted'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
