# sing2piano — the evaluation set

476 held-out real piano-cover / pop pairs (DB2 in the paper). Each pairs a piano backing track from
the [Sing2Piano](https://www.youtube.com/@sing2piano) or
[KaraoKeysPH](https://www.youtube.com/@KaraoKeysPH) channel with the song it covers. Only
original-key covers whose duration is within 5 s of the song are kept. None of these songs is in the
training corpus, and neither channel was used to train PiCoGen2.

| File | Contents |
|---|---|
| `test.csv` | `name, song_id, piano_id, instrumental_id, piano_trim`: one row per pair; ids are YouTube ids |
| `test_verified.csv` | the same rows after the original-key check: the 476 pairs used |
| `audio_manifest.json` | per downloaded file: its YouTube id, the seconds cut from its head (channel ident and leading silence), and its duration after the cut |
| `windows.csv` | the paper's five 30 s evaluation windows per pair, as start times on that trimmed timeline |
| `overlap.txt` | pairs removed because the song is also in the training corpus |

## Rebuilding it

This repository does not download or redistribute audio.

1. **Place the audio.** For every file in `audio_manifest.json`, save the YouTube upload as
   `<raw>/<video_id>.<ext>`: the piano cover, the song, and, where listed, the official
   instrumental.
2. **Trim** each file as the paper's download did, so the files of a pair start at the same
   musical moment:
   ```bash
   python sing2piano/trim.py --raw <raw>
   ```
3. **Separate and transcribe** (needs `P2PA_MIR_PYTHON` and `PICOGEN_PYTHON`, see the main README):
   ```bash
   uv run python sing2piano/process.py --stage separate   --device 0
   uv run python sing2piano/process.py --stage transcribe --device 0
   uv run python sing2piano/process.py --stage piano      --device 0
   ```

   | Stage | Output under `sing2piano_audio/` | Use |
   |---|---|---|
   | `separate` | `processed/<piano_id>.<ext>`: the official instrumental where it exists, otherwise the song through `htdemucs --two-stems vocals` | the reference pop production |
   | `transcribe` | `midi/<piano_id>.mid`: MuScriptor over that instrumental, exactly as the training corpus is transcribed | training-distribution inputs of Base, FST and RDA |
   | `piano` | `midi_piano/<piano_id>.mid`: Kong et al.'s transcriber over the real piano cover | real-piano (out-of-distribution) inputs of every system |

The training-distribution input for PiCo is a PiCoGen2 cover of the same instrumental. Generate it
with `corpus/picogen/run.py picogen` on an inventory of these instrumentals.

Each window in `windows.csv` is 30 s from `start_seconds` on the trimmed timeline of both the piano
cover and the song. The 476 pairs are not temporally verified: arrangements can restructure a song.
The paper's APA reference uses a separate, manually aligned 33-pair subset.
