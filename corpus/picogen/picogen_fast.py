"""Output-equivalent fast inference path for PiCoGen 2.

Loaded by ``picogen_worker.py`` inside the PiCoGen environment (``corpus/picogen/setup.sh``). The
upstream implementation re-encodes discarded zero conditions, rebuilds full-history CPU tensors per
token, and drops the GPT KV cache at every bar boundary. This implementation retains the same model
and sampler while feeding only the suffix absent from the cache.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def picogen2_module(name: str = ""):
    """A PiCoGen2 module: the public `picogen2` package (github.com/tanchihpin0517/PiCoGen, branch
    v2) or its copy inside a PiCoCtrl checkout, `picoctrl.picogen2`. The decoder, tokenizer and
    checkpoint are the same; the fast decoder never calls the one method (`generate`) they differ in.
    """
    import importlib

    for package in ("picogen2", "picoctrl.picogen2"):
        try:
            return importlib.import_module(f"{package}.{name}" if name else package)
        except ModuleNotFoundError as error:
            if error.name not in {package, package.split(".")[0], f"{package}.{name}"}:
                raise
    raise ModuleNotFoundError("neither picogen2 nor picoctrl.picogen2 is importable")


def _sample(model, segments, classes, encode, cache, temperature):
    """Encode a new suffix, append it to ``cache``, and sample one token."""
    top_p = picogen2_module("utils").top_p

    device = next(model.parameters()).device
    length = len(segments)
    input_ids = torch.zeros(1, length, dtype=torch.long, device=device)
    condition_embeddings = torch.zeros(
        1, length, model.hp.d_model, dtype=torch.float32, device=device
    )

    condition_positions = [position for position, value in enumerate(encode) if value]
    target_positions = [position for position, value in enumerate(encode) if not value]
    if target_positions:
        input_ids[0, target_positions] = torch.as_tensor(
            [segments[position] for position in target_positions], dtype=torch.long, device=device
        )
    if condition_positions:
        raw = torch.stack([
            torch.as_tensor(segments[position], dtype=torch.float32, device=device)
            for position in condition_positions
        ])[None, ...]
        encoded = model.cond_encoder(raw)
        condition_embeddings[0, condition_positions] = encoded[0]

    class_ids = torch.as_tensor(classes, dtype=torch.long, device=device)[None, :]
    embeddings = (
        model.word_emb(input_ids) + condition_embeddings + model.cls_emb(class_ids)
    )
    output = model.model(inputs_embeds=embeddings, past_key_values=cache)
    logits = model.lm_head(output.last_hidden_state)[:, -1, :]
    probabilities = F.softmax(top_p(logits, thres=0.9, temperature=temperature), dim=-1)
    token = torch.multinomial(probabilities, num_samples=1)
    return int(token[0, 0]), output.past_key_values


@torch.inference_mode()
def decode(
    model,
    tokenizer,
    beat_information,
    melody_last_embs,
    harmony_last_embs,
    max_bar_num=None,
    max_token_num=None,
    temperature=1.0,
    device=None,
):
    """Generate with a suffix-only cross-bar KV cache and bar-level progress."""
    PiCoGenDecoder = picogen2_module("model").PiCoGenDecoder
    Event = picogen2_module("repr").Event
    downbeat_time_to_index = picogen2_module("utils").downbeat_time_to_index

    if device is None:
        device = next(model.parameters()).device
    starting_bpm = 60 / np.diff(np.asarray(beat_information["beats"])[:2]).mean()
    target_class = PiCoGenDecoder.InputClass.TARGET.value
    condition_class = PiCoGenDecoder.InputClass.CONDITION.value
    song_start = Event(etype="spec", value="spec_ss")
    end_event = Event(etype="spec", value="spec_se")
    bar_start = Event(etype="bar", value="bar_start")
    bar_end = Event(etype="bar", value="bar_end")
    out_events = [song_start, tokenizer.get_tempo_event(starting_bpm)]
    segments = [tokenizer.e2i(Event(etype="spec", value="spec_bos"))]
    segments.extend(tokenizer.e2i(event) for event in out_events)
    classes = [target_class] * len(segments)
    encode = [False] * len(segments)

    # One transfer per song replaces a NumPy copy and host-to-device transfer per condition slot.
    melody = torch.as_tensor(melody_last_embs, dtype=torch.float32, device=device)
    harmony = torch.as_tensor(harmony_last_embs, dtype=torch.float32, device=device)
    features = torch.stack((melody, harmony), dim=1)

    total_beats = len(beat_information["beats"])
    downbeats = downbeat_time_to_index(
        beat_information["beats"], beat_information["downbeats"]
    )
    if downbeats[-1] < total_beats:
        downbeats.append(total_beats - 1)
    if max_bar_num is not None:
        downbeats = downbeats[: max_bar_num + 1]

    cache = None
    cached_length = 0
    progress = tqdm(total=len(downbeats) - 1, desc="picogen", unit="bar")
    for bar_index in range(len(downbeats) - 1):
        start, end = downbeats[bar_index], downbeats[bar_index + 1]
        for position in range(start * tokenizer.beat_div, end * tokenizer.beat_div):
            segments.append(features[position])
            classes.append(condition_class)
            encode.append(True)
        if bar_index == len(downbeats) - 2:
            segments.append(tokenizer.e2i(end_event))
            classes.append(condition_class)
            encode.append(False)
        segments.append(tokenizer.e2i(bar_start))
        classes.append(target_class)
        encode.append(False)
        out_events.append(bar_start)

        while True:
            if len(segments) > model.hp.max_seq_len:
                segments = segments[-model.hp.max_seq_len // 2 :]
                segments[0] = tokenizer.e2i(Event(etype="spec", value="spec_bos"))
                classes = classes[-model.hp.max_seq_len // 2 :]
                classes[0] = target_class
                encode = encode[-model.hp.max_seq_len // 2 :]
                encode[0] = False
                cache = None
                cached_length = 0

            token, cache = _sample(
                model,
                segments[cached_length:],
                classes[cached_length:],
                encode[cached_length:],
                cache,
                temperature,
            )
            cached_length = len(segments)
            event = tokenizer.i2e(token)
            out_events.append(event)
            segments.append(token)
            classes.append(target_class)
            encode.append(False)
            if event in (bar_end, end_event):
                break

        progress.update(1)
        progress.set_postfix(tokens=len(out_events), refresh=False)
        if max_token_num is not None and len(segments) > max_token_num:
            break
    progress.close()
    return out_events

