"""Compare the lightweight local token observer with llama-cpp's public API."""
import gc
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llama_cpp import Llama, __version__ as llama_version
from hari_ai.runtime.local_model import LocalModel, bundled_model_path


def main():
    messages = [{'role':'user', 'content':'Briefly explain why the sky looks blue.'}]
    start = time.perf_counter()
    local = LocalModel(context_size=2048)
    try:
        actual_text = ''.join(local.chat_stream(messages, max_tokens=64))
        actual = local.last_token_likelihood
        actual_seconds = time.perf_counter()-start
    finally:
        local.close()
    gc.collect()
    start = time.perf_counter()
    reference = Llama(model_path=str(bundled_model_path()), n_ctx=2048, n_threads=4,
                      chat_format='chatml', logits_all=True, verbose=False, seed=42)
    probabilities, text = [], []
    try:
        for chunk in reference.create_chat_completion(messages=messages, max_tokens=64,
                temperature=.2, top_p=.9, repeat_penalty=1.1, stream=True,
                logprobs=True, top_logprobs=1):
            choice = chunk['choices'][0]
            text.append(choice.get('delta', {}).get('content', '') or '')
            probabilities.extend(math.exp(item['logprob']) for item in
                                 ((choice.get('logprobs') or {}).get('content') or []))
        reference_seconds = time.perf_counter()-start
    finally:
        reference.close()
    expected = sum(probabilities)/len(probabilities) if probabilities else None
    print(json.dumps({'llama_cpp_python':llama_version, 'tokens':len(probabilities),
        'text_matches':actual_text == ''.join(text), 'observer_mean':actual,
        'public_api_mean':expected, 'absolute_difference':abs(actual-expected)
        if actual is not None and expected is not None else None,
        'observer_seconds':round(actual_seconds, 3), 'public_api_seconds':round(reference_seconds, 3),
        'caveat':'One local run, including model loading; timings are not a benchmark guarantee.'}, indent=2))


if __name__ == '__main__':
    main()
