"""Fit and evaluate calibration for binary answer correctness.

Calibration is conditional on the exact model, prompt protocol, and task
distribution used for evaluation. A passing development check cannot establish
perfect accuracy or transfer to unrelated questions.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Dict, Iterable, List


def scale_probability(value: float, temperature: float) -> float:
    if not 0 <= value <= 1 or not .05 <= temperature <= 20:
        raise ValueError('Probability or temperature is outside the supported range.')
    value = min(1 - 1e-7, max(1e-7, value))
    logit = math.log(value / (1 - value)) / temperature
    result = 1 / (1 + math.exp(-max(-34, min(34, logit))))
    return min(1 - 1e-15, max(1e-15, result))


def platt_probability(value: float, slope: float, intercept: float) -> float:
    if not 0 <= value <= 1 or not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError('Invalid score or calibration parameters.')
    value = min(1 - 1e-7, max(1e-7, value))
    z = max(-34, min(34, slope * math.log(value/(1-value)) + intercept))
    return min(1 - 1e-15, max(1e-15, 1/(1+math.exp(-z))))


def fit_platt(rows: Iterable[Dict[str, Any]]) -> tuple[float, float]:
    """Fit two parameters on calibration data, with mild regularization."""
    import torch
    data = list(rows)
    if not data or {int(r['label']) for r in data} != {0, 1}:
        raise ValueError('Calibration requires both correct and incorrect examples.')
    values = [float(r['probability']) for r in data]
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in values):
        raise ValueError('Calibration scores must be finite probabilities.')
    values = [min(1-1e-7, max(1e-7, p)) for p in values]
    x = torch.tensor([math.log(p/(1-p)) for p in values], dtype=torch.float64)
    y = torch.tensor([r['label'] for r in data], dtype=torch.float64)
    params = torch.nn.Parameter(torch.tensor([1.0,0.0], dtype=torch.float64))
    optimizer = torch.optim.LBFGS([params], max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(params[0]*x + params[1], y)
        loss = loss + .001*params[0].square()
        loss.backward()
        return loss
    optimizer.step(closure)
    slope, intercept = (float(p) for p in params.detach())
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError('Calibration did not converge to finite parameters.')
    return slope, intercept


def fit_temperature(rows: Iterable[Dict[str, Any]]) -> float:
    """Minimize binary negative log likelihood on calibration data only."""
    data = list(rows)
    if not data or {int(r['label']) for r in data} != {0, 1}:
        raise ValueError('Calibration requires both correct and incorrect examples.')
    def loss(temp: float) -> float:
        total = 0.0
        for row in data:
            p = scale_probability(float(row['probability']), temp)
            total -= row['label'] * math.log(p) + (1-row['label']) * math.log1p(-p)
        return total / len(data)
    return min((.05 * (400 ** (index / 300)) for index in range(301)), key=loss)


def wilson_accuracy_interval(correct: int, count: int) -> tuple[float, float]:
    """Wilson 95% binomial interval, assuming independent representative rows."""
    if (not isinstance(correct, int) or not isinstance(count, int) or
            count <= 0 or correct < 0 or correct > count):
        raise ValueError('Expected a valid number of correct outcomes and trials.')
    z = 1.959963984540054
    p = correct/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return max(0.0, center-radius), min(1.0, center+radius)


def calibration_metrics(rows: Iterable[Dict[str, Any]], temperature: float = 1.0,
                        slope: float | None = None, intercept: float = 0.0) -> Dict[str, Any]:
    data = list(rows)
    if not data:
        raise ValueError('Evaluation requires examples.')
    counts = Counter(int(row['label']) for row in data)
    values = [(platt_probability(float(r['probability']), slope, intercept) if slope is not None
               else scale_probability(float(r['probability']), temperature), int(r['label'])) for r in data]
    positive = [p for p,y in values if y == 1]
    negative = [p for p,y in values if y == 0]
    auc = (sum((p > q) + .5*(p == q) for p in positive for q in negative)
           / (len(positive)*len(negative))) if positive and negative else None
    bins: List[Dict[str, Any]] = []
    ece = 0.0
    for index in range(10):
        group = [(p, y) for p, y in values if min(9, int(p*10)) == index]
        if group:
            mean_p = sum(p for p, _ in group) / len(group)
            correct = sum(y for _, y in group)
            accuracy = correct / len(group)
            lower, upper = wilson_accuracy_interval(correct, len(group))
            ece += len(group) / len(values) * abs(mean_p-accuracy)
            bins.append({'index':index, 'count':len(group), 'confidence':mean_p, 'accuracy':accuracy,
                         'accuracy_interval_95':[lower, upper]})
    return {'count':len(values), 'correct':counts[1], 'incorrect':counts[0],
            'brier':sum((p-y)**2 for p,y in values)/len(values), 'ece_10':ece,
            'auroc':auc, 'bins':bins}


def brier_improvement_interval(rows: Iterable[Dict[str, Any]], *, baseline: float,
                               slope: float, intercept: float, samples: int = 1000,
                               seed: int = 1108) -> Dict[str, float]:
    """Paired bootstrap of held-out Brier gain over a fixed baseline.

    Each row must represent one independent problem group. The interval holds
    the fitted calibrator and baseline fixed; it does not include uncertainty
    from training or calibration fitting, and assumes representative groups.
    """
    data = list(rows)
    if len(data) < 2 or samples < 100 or not 0 <= baseline <= 1:
        raise ValueError('Bootstrap requires valid data, baseline, and at least 100 resamples.')
    groups = [r.get('group_id') for r in data]
    if any(not group for group in groups) or len(set(groups)) != len(groups):
        raise ValueError('Each bootstrap row must have an independent group id.')
    gains = []
    for row in data:
        if row.get('label') not in (0, 1):
            raise ValueError('Bootstrap labels must be binary.')
        label = int(row['label'])
        probability = platt_probability(float(row['probability']), slope, intercept)
        gains.append((baseline-label)**2 - (probability-label)**2)
    rng = random.Random(seed)
    n = len(gains)
    estimates = sorted(sum(gains[rng.randrange(n)] for _ in range(n))/n for _ in range(samples))
    return {'estimate':sum(gains)/n, 'lower_95':estimates[int(samples*.025)],
            'upper_95':estimates[min(samples-1, int(samples*.975))],
            'resamples':samples}


def evaluate_calibrator(calibration: List[Dict[str, Any]], test: List[Dict[str, Any]], *,
                        predeclared: bool) -> Dict[str, Any]:
    """Fit on one split; report and gate on a disjoint predeclared test split."""
    for row in calibration + test:
        if row.get('label') not in (0, 1) or not row.get('group_id'):
            raise ValueError('Each row requires a binary label and a problem group_id.')
    ids = [r['group_id'] for r in calibration + test]
    if len(set(ids)) != len(ids):
        raise ValueError('Every problem group must occur only once.')
    if {r['group_id'] for r in calibration} & {r['group_id'] for r in test}:
        raise ValueError('Calibration and test problem groups overlap.')
    slope, intercept = fit_platt(calibration)
    fit = calibration_metrics(calibration, slope=slope, intercept=intercept)
    heldout = calibration_metrics(test, slope=slope, intercept=intercept)
    raw = calibration_metrics(test)
    rate = fit['correct'] / fit['count']
    baseline_brier = sum((rate-r['label'])**2 for r in test) / len(test)
    interval = brier_improvement_interval(test, baseline=rate, slope=slope, intercept=intercept)
    # This is deliberately a minimum evidence gate, not proof of calibration
    # outside this distribution. Do not tune these gates on the test set.
    issues = []
    if not predeclared:
        issues.append('test split was previously explored')
    for label, metrics in [('calibration', fit), ('test', heldout)]:
        if metrics['count'] < 100 or min(metrics['correct'], metrics['incorrect']) < 20:
            issues.append(f'{label} needs at least 100 examples and 20 of each class')
    if heldout['ece_10'] > .10:
        issues.append('held-out ECE exceeds 0.10')
    if heldout['brier'] >= baseline_brier:
        issues.append('held-out Brier does not beat the constant baseline')
    if interval['lower_95'] <= 0:
        issues.append('held-out Brier improvement is not supported by the bootstrap interval')
    if heldout['auroc'] is None or heldout['auroc'] < .60:
        issues.append('held-out AUROC is below 0.60')
    return {'method':'platt_scaling', 'slope':slope, 'intercept':intercept,
            'calibration':fit, 'test':heldout, 'test_before_scaling':raw,
            'constant_baseline_brier':baseline_brier, 'brier_improvement_interval':interval,
            'predeclared_split':predeclared,
            'eligible':not issues, 'issues':issues}
