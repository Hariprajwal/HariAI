"""Compare independent HariNano architecture parameter and size budgets."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hari_ai.model.budget import ModelBudget


def main():
    candidates = {
        'compact_research': ModelBudget(16384, 512, 30, 8, 2, 1536),
        'target_125m': ModelBudget(16384, 576, 32, 9, 3, 1600),
        'wider_alternative': ModelBudget(16384, 640, 28, 10, 2, 1536),
        'smaller_deployment_study': ModelBudget(16384, 512, 22, 8, 2, 1376),
    }
    output = {}
    for name, cfg in candidates.items():
        output[name] = {'config':asdict(cfg), 'parameters':cfg.breakdown(),
                        'ideal_fp32_bytes':cfg.ideal_weight_bytes(32),
                        'ideal_fp16_bytes':cfg.ideal_weight_bytes(16),
                        'ideal_q4_bytes':cfg.ideal_weight_bytes(4)}
    print(json.dumps({'warning':'Q4 sizes are lower bounds; real formats require metadata and scales.',
                      'candidates':output}, indent=2))


if __name__ == '__main__':
    main()
