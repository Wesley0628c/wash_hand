"""Reproducible clip-labelled evaluation; cached detections isolate logic changes.

Labels come from clip filenames, not adjacent screenshots. All frames (including
transitions) are included. These samples are development data, not a held-out test.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.pipeline import WashHandPipeline, PipelineConfig
from src.evaluation_metrics import frame_metrics

STEPS = dict(zip('內外夾弓大立腕', ['inside', 'outside', 'interlace', 'knuckles', 'thumb', 'fingertips', 'wrist']))


def evaluate(cache_path, output):
    cache_path, output = Path(cache_path), Path(output)
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else None
    recordings, summaries, rows = [], [], []
    for group in ('sample_v1', 'sample_v2'):
        for zh, target in STEPS.items():
            path = Path('data') / group / f'{zh}.mov'
            clip = next((c for c in cached if c['path'] == str(path)), None) if cached else None
            class ReplayDetector:
                last_metadata = {}
                def close(self):
                    pass
            pipeline = WashHandPipeline(PipelineConfig(), detector=ReplayDetector() if clip is not None else None)
            if clip is None:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    raise RuntimeError(f'Cannot open {path}')
                fps = cap.get(cv2.CAP_PROP_FPS)
                clip = {'path': str(path), 'fps': fps, 'frames': []}
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    left, right, _ = pipeline.detector.process(frame)
                    clip['frames'].append({'left': None if left is None else left.tolist(),
                                           'right': None if right is None else right.tolist(),
                                           'meta': dict(pipeline.detector.last_metadata)})
                cap.release()
                pipeline.detector.reset()
            recordings.append(clip)
            counts, raw_counts, credit = Counter(), Counter(), Counter()
            for i, entry in enumerate(clip['frames']):
                def replay(frame, crop_split_screen=None):
                    pipeline.detector.last_metadata = entry['meta']
                    return tuple(None if entry[k] is None else np.array(entry[k], dtype=np.float32) for k in ('left', 'right')) + (None,)
                pipeline.detector.process = replay
                res, _ = pipeline.process_frame(np.zeros((1, 1, 3), np.uint8), (i + 1) / clip['fps'])
                counts[res.display_label] += 1
                raw_counts[res.raw_label] += 1
                credited = getattr(res, 'credit_label', res.display_label)
                credit[credited] += res.quality_score / clip['fps']
                rows.append({'clip': str(path), 'target': target, 'time': res.timestamp_sec,
                             'raw': res.raw_label, 'display': res.display_label,
                             'credit': credited, 'weight': res.quality_score,
                             'hands': res.num_hands_observed, 'confidence': res.confidence,
                             'reason': getattr(res, 'evidence_reason', ''),
                             'back_score': res.features.get('back_evidence', {}).get('score', 0),
                             'back_role': res.features.get('back_evidence', {}).get('role', ''),
                             'back_motion_seconds': res.features.get('back_evidence', {}).get('motion_seconds', 0),
                             'back_credit_ready': res.features.get('back_evidence', {}).get('credit_ready', False),
                             'completed': res.just_completed_step or ''})
            summary = {'clip': str(path), 'target': target, 'frames': len(clip['frames']),
                       'display_counts': dict(counts), 'raw_counts': dict(raw_counts),
                       'correct_fraction': counts[target] / max(1, sum(counts.values())),
                       'wrong_fraction': sum(v for k, v in counts.items() if k not in (target, 'other')) / max(1, sum(counts.values())),
                       'credit_seconds': dict(credit),
                       'completed': sorted(pipeline.state_machine.completed_steps)}
            summaries.append(summary)
            pipeline.detector.close()
            print(path, round(summary['correct_fraction'], 3), summary['completed'], flush=True)
    if cached is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(recordings))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    with output.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix('.metrics.json').write_text(json.dumps(frame_metrics(rows), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='scratch/sample_detections.json')
    parser.add_argument('--output', default='docs/evaluation/samples')
    args = parser.parse_args()
    evaluate(args.cache, args.output)
