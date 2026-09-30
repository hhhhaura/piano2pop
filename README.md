# Piano2Pop

Code for **Piano2Pop: Expanding Piano Demos into Pop Productions Without Aligned Supervision**
(Yun-Chen Cheng, Chih-Pin Tan, Tzu-Hung Huang, Yi-Hsuan Yang; submitted to ICASSP 2027).

**Demo:** https://hhhhaura.github.io/piano2pop/

Piano2Pop turns a piano MIDI demo into an instrumental pop production. It fine-tunes ACE-Step v1.5
(a 2.39B rectified-flow transformer) with a piano-roll encoder, trained on *pseudo demos*: piano MIDI
derived from pop instrumentals, so no aligned piano–pop pairs are needed.

| Paper | Setting | Pseudo-demo source |
|---|---|---|
| Base | `base` | MuScriptor transcription of the full instrumental, restricted to piano |
| FST  | `var`  | per four-bar segment: the Base transcription (0.4) or a piano transcription of one of four stem mixtures — Full, Harmonic, No-bass, Piano+bass (0.15 each) |
| RDA  | `rule` | the Base transcription with register–density augmentation: function-aware note deletion (0.2) or octave folding (0.3), else unchanged (0.5) |
| PiCo | `pico` | PiCoGen2-generated piano covers, beat-aligned to the instrumental |

`final` (PiCoGen, Base and the stem-mixture transcriptions together) is included as well.

## Model

- **Backbone:** `ACE-Step/acestep-v15-base`, fully fine-tuned; the frozen ACE-Step VAE maps 48 kHz
  stereo audio to 64-channel latents at 25 Hz.
- **Piano-roll encoder:** 176 × T at 25 Hz (pitches 21–108; binary sustain plane, onset plane
  carrying velocity / 127) through a five-layer residual 1-D convolution (kernel 5, width 512,
  GroupNorm, SiLU). Its output replaces ACE-Step's reference-audio condition and is concatenated
  with the all-ones generation mask and the noisy latent (192 channels). The output projection is
  zero-initialised on top of ACE-Step's silence latent, so an untrained model is stock ACE-Step.
- **Text:** every example uses the same fixed prompt, *"instrumental, no vocals"*; the piano is the
  only song-specific condition.
- **Training:** 500,000 steps, batch 4, one GPU. The encoder trains alone for the first 2,000 steps,
  then jointly with the DiT (learning rates 1e-4 and 1e-5). Windows start at 20 s for 100,000 steps
  and grow by 5 s every 10,000 steps to 30 s.

## Setup

[`uv`](https://docs.astral.sh/uv/) is the only prerequisite; it also manages Python. `ffmpeg` is
required.

```bash
P2PA_CUDA=cu124 P2PA_DATA_ROOT=/path/to/data bash scripts/setup.sh   # or P2PA_CUDA=cu130
source env.sh
```

`scripts/setup.sh` installs the project and writes `env.sh`. `$P2PA_DATA_ROOT` holds everything
generated (corpus, caches, runs), so it should sit on a large disk. Pick the CUDA build that matches
your driver: `cu124` for CUDA 12.4 drivers, `cu130` for recent GPUs. `bash scripts/fetch_models.sh`
caches the ACE-Step weights, so that later runs can work offline (`HF_HUB_OFFLINE=1`).

## Building the training corpus

The corpus is the paper's 8,876 pop songs, listed with their YouTube ids and splits in
[`corpus/tracks.csv`](corpus/tracks.csv): 8,463 for training and 413 held out for validation
(recorded as 243 `validation` and 170 `test`). This repository does not download or
redistribute audio.

1. **Place the audio.** For each row, save the song's audio (the full mix, as uploaded) as
   `$P2PA_DATA_ROOT/data/raw/<youtube_id>.<ext>` (m4a, mp3, webm, opus, wav or flac). Tracks you
   cannot obtain are skipped.
2. **Corpus tools** (HTDemucs and MuScriptor, in their own environment; MuScriptor's weights are
   gated on Hugging Face, so export an `HF_TOKEN` with access first):
   ```bash
   bash scripts/setup_corpus_tools.sh       # then add the printed P2PA_MIR_PYTHON to env.sh
   ```
3. **[PiCoGen2](https://github.com/tanchihpin0517/PiCoGen/tree/v2) and BeatThis**, in their own
   Python 3.11 environment (the script clones PiCoGen2 and downloads its, SheetSage's and Jukebox's
   weights, about 10 GB):
   ```bash
   bash corpus/picogen/setup.sh      # add the printed exports to env.sh
   ```
4. **Run every stage**, one shard per GPU:
   ```bash
   GPUS=0,1,2,3 bash scripts/build_corpus.sh
   ```

`scripts/build_corpus.sh` runs, resumably:

| Stage | Command | Output under `$P2PA_DATA_ROOT/data/p2pdata/` |
|---|---|---|
| index | `p2pa-corpus-index` | `.cache/manifests/p2pdata.inventory.jsonl` |
| separate | `p2pa-corpus-process separate` | `audio/<shard>/<id>/`: HTDemucs vocals pass → `instrumental.mp3` (the training target); `htdemucs_6s` on it → drums, bass, other, guitar, piano |
| transcribe | `p2pa-corpus-process transcribe` | `midi/<shard>/<id>/full-piano.mid` (Base) and `{full6,harmonic,nobass,pianobass}-piano.mid` (FST); MuScriptor restricted to program 0, velocities from onset-energy rank quantiles |
| beats | `corpus/picogen/run.py beats` | `beats/<shard>/<id>.json`: BeatThis beats and downbeats, for the four-bar segments |
| picogen | `corpus/picogen/run.py picogen` | `midi/<shard>/<id>/full-picogen.mid`: PiCoGen2 covers, each generated bar stretched onto the detected bar (skip with `PICOGEN=0` if you do not train `pico`/`final`) |
| prep | `p2pa-prep`, `p2pa-prep-text` | `.cache/`: the training manifest (with the listed splits), outlier screening, whole-track VAE latents, and the fixed text prompt |

## Training

```bash
run/train.sh var                     # FST; writes runs/var under $P2PA_DATA_ROOT
P2PA_DEVICE=1 run/train.sh pico      # another setting on another GPU
run/dryrun.sh base                   # 20 steps: peak VRAM, host memory and ms/step, nothing written
```

One run is one setting on one GPU; a full fine-tune at batch 4 fits an 80 GB card with gradient
checkpointing. Runs resume from their own `last.ckpt`, and refuse to resume under a different
configuration. Any key in `configs/` can be overridden on the command line, e.g.
`run/train.sh rule seed=1 exp_name=rule_seed1`.

### From-scratch backbone (not in the paper)

`backbone=scratch` trains the same task without a pretrained model: ACE-Step's DiT is replaced by
a rectified-flow transformer ([`src/p2pa/flowmatching.py`](src/p2pa/flowmatching.py); 8 DiT
blocks, width 768, 12 heads; 220M parameters, plus the 5.7M roll encoder), trained from random
initialisation in the frozen ACE-Step VAE's latent space. Corpus, windows, curriculum, roll
encoder and all five settings are shared with the finetune.

```bash
run/train.sh pico backbone=scratch   # runs/scratch_pico
```

What differs (`configs/backbone/scratch.yaml` and `scratch:` in `configs/config.yaml`): the roll is
summed into the transformer's residual stream instead of entering `src_latents`; no text
condition; batch 16 (about 12 GB of VRAM at 30 s); one learning rate of 1e-4 for both parameter groups and
no freeze window; an EMA of the weights (decay 0.999) that validation and sampling use; the roll is
dropped with probability 0.1 in training, and sampling uses 50 Euler steps with linear guidance 2.0
on it. Its checkpoints are not released.

## Generating

```bash
uv run p2pa-sample $P2PA_DATA_ROOT/runs/var/checkpoints/last.ckpt --midi demo.mid --output out/
uv run p2pa-sample ... --mode glued_full_song      # a whole song, as overlapping 30 s windows
```

The same command samples a from-scratch checkpoint; its backbone, step count and guidance are read
from the checkpoint.

The piano MIDI can be any piano performance or transcription (for real piano recordings, the paper
uses Kong et al.'s transcriber). Output is `out/generated.mp3`, with the conditioning window as
`piano.mid`.

## Evaluation set

[`sing2piano/`](sing2piano/) builds the 476 held-out piano-cover / pop pairs from the Sing2Piano and
KaraoKeysPH channels (`sing2piano/test.csv`): separation of each song's instrumental, MuScriptor
transcriptions of it (training-distribution inputs) and Kong transcriptions of the real piano covers
(real-world inputs). As with the corpus, you place the audio yourself.

## Checkpoints

The finetuned checkpoints are to be released.

## Repository layout

```
src/p2pa/        model, piano-roll encoder, dataset and augmentation, training, sampling, corpus stages;
                 scratch.py, flow.py and flowmatching.py for the from-scratch backbone
configs/         config.yaml (the paper's values), setting/{base,var,rule,pico,final}.yaml, and
                 backbone/{acestep,scratch}.yaml
corpus/          tracks.csv, and picogen/ (PiCoGen2 and BeatThis workers, driver, environment setup)
sing2piano/      the evaluation set
run/             training launchers
scripts/         setup, model download and the corpus build
tests/           pytest suite, including a check that every setting reproduces its trained run's config
```

## Citation

```bibtex
@misc{cheng2026piano2pop,
  title  = {Piano2Pop: Expanding Piano Demos into Pop Productions Without Aligned Supervision},
  author = {Cheng, Yun-Chen and Tan, Chih-Pin and Huang, Tzu-Hung and Yang, Yi-Hsuan},
  note   = {Submitted to ICASSP 2027},
  year   = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). It builds on ACE-Step v1.5 (MIT), MuScriptor,
HTDemucs, BeatThis, PiCoGen2 and Kong et al.'s piano transcriber, each under its own license. The
corpus is a list of public YouTube ids; the audio remains its owners'.
