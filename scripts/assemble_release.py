"""Assemble only evaluated candidates into the standalone local package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--operation', type=Path, default=Path('checkpoints/operation_jev'))
    parser.add_argument('--fresh-report', type=Path, required=True)
    args = parser.parse_args()
    language = json.loads((args.candidate/'report.json').read_text(encoding='utf-8'))
    operation = json.loads((args.operation/'report.json').read_text(encoding='utf-8'))
    fresh = json.loads(args.fresh_report.read_text(encoding='utf-8'))
    model = args.candidate/'hariai-chat.gguf'
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    if model_hash != language['candidate_sha256'] or model_hash != fresh['candidate']['model_sha256']:
        raise ValueError('Evaluation hashes do not match the candidate.')
    if not language['narrow_gates_passed'] or not operation['passed']:
        raise ValueError('Candidate failed development gates; not promoted.')
    if fresh['candidate']['exact_pass'] < fresh['baseline']['exact_pass']:
        raise ValueError('Fresh smoke checks regressed; not promoted.')
    root = Path(__file__).resolve().parents[1]
    assets = root/'hari_ai/assets'
    assets.mkdir(parents=True, exist_ok=True)
    for source, target in [(model, 'hariai-chat.gguf'), (args.candidate/'report.json', 'adaptation_report.json'),
                           (args.fresh_report, 'fresh_report.json'), (args.operation/'operation_jev.pt', 'operation_jev.pt'),
                           (args.operation/'report.json', 'operation_report.json'), (args.operation/'split.json', 'operation_split.json')]:
        shutil.copyfile(source, assets/target)
    files = []
    for name in ['hariai-chat.gguf', 'operation_jev.pt']:
        path = assets/name
        files.append({'name':name, 'bytes':path.stat().st_size, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    import sys
    sys.path.insert(0, str(root))
    from hari_ai.version import __version__
    manifest = {'version':__version__, 'creator':'K R Hari Prajwal', 'architecture':'Two-component bundle; not merged incompatible architectures',
        'language_foundation':language['base'], 'base_sha256':language['base_sha256'],
        'language_adaptation':'Merged LoRA identity adaptation', 'files':files,
        'total_model_bytes':sum(f['bytes'] for f in files), 'runtime_service_required':False,
        'chatgpt_equivalence_established':False, 'under_35_mb':False}
    (assets/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    card = f'''# HariAI local model bundle

Creator and adaptation author: **K R Hari Prajwal**. Experimental alpha, 8 November 2026.

## What it contains

The language file is a merged fine-tune of **{language['base']}**, originally
published by Hugging Face. It is not trained from scratch. The tokenizer and
all required language weights are in the GGUF file. Inference requires neither
Ollama nor an external model service nor a separate base-model download.

Open-JEV is packaged as a second, separately trained structured-decision model.
The two architectures have not been fused into one neural network. Its typed
readout does not improve conversation or verify arbitrary generated text.

| Artifact | Actual bytes |
| --- | ---: |
| Language GGUF | {files[0]['bytes']:,} |
| Operation-policy checkpoint | {files[1]['bytes']:,} |
| Combined neural files | {manifest['total_model_bytes']:,} |

These sizes exclude Python dependencies, installation overhead, and runtime RAM.
The original 35 MB budget is not achieved. Checksums are in `manifest.json`.

## Language adaptation

{language['optimizer_steps']} optimizer updates trained {language['trainable_parameters']:,}
LoRA parameters on 36 original examples with instructional replay. Modified
attention matrices were merged and exported with
`{language.get('changed_tensor_precision', 'original')}` precision; other tensors remain
unchanged. All inference uses the merged file, not runtime LoRA modules.

Development identity probes: {language['candidate_probe']['identity_pass']}/4.
General development substring probes: {language['candidate_probe']['general_pass']}/8,
versus {language['baseline_probe']['general_pass']}/8 for the foundation.
Development validation loss: {language['baseline_validation_loss']:.4f} to
{language['candidate_validation_loss']:.4f} nats on eight examples before export.

The fresh exact-answer smoke set scored {fresh['candidate']['exact_pass']}/12,
versus {fresh['baseline']['exact_pass']}/12 for the foundation. Eight additional
qualitative responses are retained in `fresh_report.json`. These are tiny local
probes, not a broad benchmark. The development probes were inspected during
iteration and must not be described as an independent final evaluation.

## Open-JEV experiment

The reference architecture is by Kye Gomez and upstream contributors under
Apache 2.0. HariAI trained 400 updates on 72 synthetic operation records and
evaluated 24 held-out combinations: {round(operation['test_accuracy']*24)}/24 correct.
The majority baseline is 12/24. The exact deterministic teacher rule is already
a complete solution to this finite problem. A neural model offers no demonstrated
advantage over that rule.

Supported inputs are the documented operation/scope/authorized/backup schema.
The model is advisory only and never performs an operation. It has known errors,
including an incorrect `proceed` prediction. The confidence head is untrained
and omitted from the bounded API. Probabilities are not calibrated confidence.

## Limits and attribution

The model can hallucinate, repeat itself, fail basic arithmetic, produce faulty
code, or fail instructions. No ChatGPT equivalence, broad reasoning reliability,
automatic self-improvement, or general answer calibration has been established.
Changing the creator's name does not establish broader intelligence.

Upstream SmolLM weights retain Apache 2.0 (`LICENSE-SMOLLM`). HariAI-specific
modifications follow the root HariAI Noncommercial License. The commercial
restriction does not revoke upstream rights. Preserve all attribution and
license files when redistributing. See `THIRD_PARTY_NOTICES.md` in the source.
'''
    (assets/'MODEL_CARD.md').write_text(card,encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
