"""Analyze explicitly stated integer results without marking vague text wrong."""
import argparse
import collections
import json
import re
from pathlib import Path


def extract_result(answer):
    candidates = [int(m.group(1)) for m in re.finditer(
        r'(?:=|\banswer\s+(?:is|:)|\bresult\s+(?:is|:))\s*([+-]?\d+)(?!\d|\.\d|,\d|\s*[+*/-]\s*\d)', answer, re.I)]
    match = re.fullmatch(r'([+-]?\d+)\.?', answer.strip())
    if match:
        candidates.append(int(match.group(1)))
    distinct = set(candidates)
    return next(iter(distinct)) if len(distinct) == 1 else None


def analyze(rows):
    extracted = []
    for row in rows:
        value = extract_result(row['answer'])
        if value is not None:
            extracted.append({'split': row['split'], 'correct': int(value == row['expected']),
                              'token_likelihood': row['estimate']})
    counts = collections.Counter((row['split'], row['correct']) for row in extracted)
    train = [r for r in extracted if r['split'] == 'calibration']
    test = [r for r in extracted if r['split'] == 'test']
    return {'examples':len(rows), 'unambiguously_extracted':len(extracted),
        'not_extracted':len(rows)-len(extracted), 'correct_among_extracted':sum(r['correct'] for r in extracted),
        'calibration_correct':counts[('calibration', 1)], 'calibration_incorrect':counts[('calibration', 0)],
        'test_correct':counts[('test', 1)], 'test_incorrect':counts[('test', 0)],
        'mean_token_likelihood_correct':sum(r['token_likelihood'] for r in extracted if r['correct']) / max(1,sum(r['correct'] for r in extracted)),
        'mean_token_likelihood_incorrect':sum(r['token_likelihood'] for r in extracted if not r['correct']) / max(1,sum(not r['correct'] for r in extracted)),
        'enough_labels_to_fit_minimum_5_per_class':len(train) > 0 and min(sum(r['correct'] for r in train),sum(not r['correct'] for r in train)) >= 5,
        'limitations':'Simple parser of explicit integer answers; excludes contradictory and unrecognized text. Does not grade an entire answer or general chat. Tiny and selective sample.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    print(json.dumps(analyze(report['rows']), indent=2))


if __name__ == '__main__':
    main()
