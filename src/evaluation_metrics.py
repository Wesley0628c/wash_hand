"""Frame metrics for filename-labelled development clips, including abstentions."""
import argparse
import csv
import json
from pathlib import Path

LABELS = ['other','inside','outside','interlace','knuckles','thumb','fingertips','wrist']


def frame_metrics(rows):
    result = {'label_source': 'clip_filename_including_transitions',
              'independent_test_set': False, 'labels': LABELS, 'groups': {}}
    groups = sorted({Path(r['clip']).parent.name for r in rows})
    for group in ['all'] + groups:
        selected = [r for r in rows if group == 'all' or Path(r['clip']).parent.name == group]
        channels = {}
        for channel in ('raw', 'display', 'credit'):
            matrix = {truth: {pred: 0 for pred in LABELS} for truth in LABELS}
            for row in selected:
                matrix[row['target']][row[channel]] += 1
            classes = {}
            for label in LABELS:
                tp = matrix[label][label]
                support = sum(matrix[label].values())
                predicted = sum(matrix[t][label] for t in LABELS)
                classes[label] = dict(tp=tp, support=support, predicted=predicted,
                                      precision=tp/predicted if predicted else None,
                                      recall=tp/support if support else None,
                                      f1=2*tp/(support+predicted) if support+predicted else None)
            n = len(selected)
            correct = sum(matrix[k][k] for k in LABELS)
            other = sum(matrix[k]['other'] for k in LABELS)
            channels[channel] = dict(frames=n, correct=correct, other=other,
                                     wrong=n-correct-other, classes=classes,
                                     confusion_matrix=matrix)
        result['groups'][group] = channels
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--output')
    args = parser.parse_args()
    with open(args.csv) as f:
        metrics = frame_metrics(list(csv.DictReader(f)))
    output = Path(args.output) if args.output else Path(args.csv).with_suffix('.metrics.json')
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
