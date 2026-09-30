"""Shared pieces for the sing2piano set: paths, JSONL checkpoints, and yt-dlp access.

The collection idioms here are lifted from `p2p/experiment/pop2piano/` rather than reinvented —
`title_key` is that project's `harvest.py:104` verbatim in behaviour, and `is_transient` encodes
the same hard-won rule: a transient failure must never reach a resume log, because recording one
permanently retires a good song.

Nothing in this package imports `p2pa` or `evaluation`. It is a data-collection tool that happens
to live in the repo, so it stays runnable with only yt-dlp present:

    uv run --with yt-dlp python sing2piano/enumerate.py
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
LISTEN = HERE / "listen"

@dataclass(frozen=True)
class Channel:
    """One karaoke channel: where it lives, how it names things, and what it prepends.

    Every field except `url` exists because the two channels collected so far disagree about it.
    Adding a third means adding a row here, not editing the stages.
    """

    name: str
    url: str
    # Dropped outright. A medley has no single source recording, and a slowed or abridged
    # arrangement is not the take the commercial recording is.
    reject: str
    # The trailing brand, stripped before anything else is read. It may swallow a leading " - ",
    # which is what turns KaraoKeysPH's `(Lower Key - Piano Karaoke)` into a bare `(Lower Key)`
    # that the shared key pattern can then read.
    brand: str
    # Seconds of channel ident before the music starts. Subtracted from the piano's duration
    # before any comparison, and trimmed from the audio on download.
    intro_seconds: float = 0.0

    @property
    def cache(self) -> Path:
        return CACHE / self.name


CHANNELS: dict[str, Channel] = {
    "sing2piano": Channel(
        name="sing2piano",
        url="https://www.youtube.com/channel/UCIk6z4gxI5ADYK7HmNiJvNg/videos",
        reject=r"guess who|cover challenge|pick your favou?rite|which version|announcement"
               r"|\bshorts?\b|subscribe|thank you for|behind the scenes|q&a|\bmedley\b",
        brand=r"[\(\[]?\s*(?:piano\s+karaoke|karaoke\s+piano|piano\s+instrumental)"
              r"(?:\s+version)?(?:\s+with\s+lyrics?)?\s*[\)\]]?",
    ),
    "karaokeys": Channel(
        name="karaokeys",
        url="https://www.youtube.com/@KaraoKeysPH/videos",
        # 83 medleys and throwback compilations, and the slowed and abridged re-cuts.
        reject=r"\bmedley\b|\bthrowback\b|\bnonstop\b|songs for|best of|sing along"
               r"|\bslow piano karaoke\b|\bshort piano karaoke\b|it's been a great year",
        brand=r"(?:\s*-\s*)?(?:hd\s+|slow\s+|short\s+)?piano\s+karaoke"
              r"(?:\s+with\s+lyrics?)?|(?:\s*-\s*)?piano\s+backing\s+track",
        # Measured, not assumed: across five sampled uploads the ident occupies 0.55-5.27 s
        # with millisecond-level consistency, so it is a fixed asset rather than a per-video
        # recording. 5.0 leaves a sliver of its decay; 5.3 clears it on every sample.
        #
        # The music itself does not begin until 7.3-8.9 s, and that gap *does* vary per upload.
        # So this removes the ident but not the leading silence: durations here still run a few
        # seconds longer than the music they contain.
        intro_seconds=5.3,
    ),
}
DEFAULT_CHANNEL = "sing2piano"


def channel(name: str) -> Channel:
    if name not in CHANNELS:
        raise SystemExit(f"unknown channel {name!r}; known: {', '.join(sorted(CHANNELS))}")
    return CHANNELS[name]
TEST_CSV = HERE / "test.csv"
OVERLAP_TXT = HERE / "overlap.txt"

# The training corpus, for the overlap report only: `$SING2PIANO_CORPUS` names the directory
# holding p2pdata/, pop2piano/ and p2pa_cache/. `overlap.py` says so if it is unset.
CORPUS_ROOT = Path(os.environ.get("SING2PIANO_CORPUS", "").strip() or "SING2PIANO_CORPUS-is-unset")
FETCH_LOG = CORPUS_ROOT / "pop2piano" / "fetch.jsonl"
MANIFEST = CORPUS_ROOT / "p2pa_cache" / "manifests" / "p2pdata.jsonl"
KONG_COVERS = CORPUS_ROOT / "p2p_cache" / "evaluation" / "kong_songeval"
# The curated panel, which is a strict subset of what those JSONLs recorded attempts for.
KONG_CURATED = HERE.parent / "corpus" / "kong_cover_ids.txt"

# Cookies for YouTube's "confirm you're not a bot" wall, in Netscape format. Optional; only needed
# if a run starts failing with `login_required`.
COOKIES = Path(os.environ.get("YTDLP_COOKIES", "").strip() or "no-cookies")

MIN_SECONDS = 60.0
MAX_SECONDS = 900.0

# Only boilerplate is stripped, never bracketed text: Mandarin MV titles put the song name inside
# the brackets, so stripping them collapses an artist's discography into one key. This is
# `harvest.py`'s rule and the corpus titles were normalised under it, so the overlap report has to
# use the same one to compare like with like.
_BOILERPLATE = re.compile(
    r"official (music )?(video|mv|audio|lyric video)|official video|official mv|lyric video"
    r"|music video|remastered|\bmv\b|\bhd\b|\b4k\b|\bofficial\b",
    re.IGNORECASE,
)


def title_key(title: str) -> str:
    """A song's identity for matching: its title minus upload boilerplate and punctuation."""
    return re.sub(r"[^0-9a-z一-鿿]+", "", _BOILERPLATE.sub(" ", title).lower())


def slug(value: str) -> str:
    """`Blank Space` -> `blank_space`. Alphanumerics and CJK survive; everything else joins."""
    out = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "_", value.lower())
    return out.strip("_")


def ydl(**overrides):
    """A metadata-only yt-dlp. `extract_flat` keeps a 2,974-video channel to one cheap call."""
    import yt_dlp

    options = {
        "quiet": True, "no_warnings": True, "skip_download": True,
        "extract_flat": "in_playlist", "ignoreerrors": True,
        "retries": 2, "socket_timeout": 30,
    }
    if COOKIES.is_file():
        options["cookiefile"] = str(COOKIES)
    options.update(overrides)
    return yt_dlp.YoutubeDL(options)


# A block, a throttle or a timeout says nothing about the video. Writing it to a resume log
# retires a good song forever; p2p burned 3,755 songs learning this.
_TRANSIENT = re.compile(
    r"sign in to confirm|not a bot|too many requests|rate.?limit|429|timed? ?out|timeout"
    r"|temporary|connection reset|unable to download|network|throttl",
    re.IGNORECASE,
)


def is_transient(error: str) -> bool:
    return bool(_TRANSIENT.search(error))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomic, so an interrupted stage leaves the previous checkpoint intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            for row in rows))
    os.replace(temp, path)
