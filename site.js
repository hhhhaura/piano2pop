(function () {
  "use strict";
  // YouTube's IFrame API, loaded once and only when someone first opens a reference video, so a
  // visitor who never clicks one makes no request to YouTube at all.
  let youtubeReady = null;
  function loadYouTube() {
    if (!youtubeReady) {
      youtubeReady = new Promise(resolve => {
        window.onYouTubeIframeAPIReady = () => resolve(window.YT);
        const script = document.createElement("script");
        script.src = "https://www.youtube.com/iframe_api";
        document.head.append(script);
      });
    }
    return youtubeReady;
  }
  const players = [];
  function pauseAll(except) {
    for (const audio of document.querySelectorAll("audio")) if (audio !== except) audio.pause();
    for (const player of players) {
      if (player !== except && player.pauseVideo) player.pauseVideo();
    }
  }
  const clock = seconds => {
    const whole = Math.floor(seconds);
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
  };

  function reference(ref, title) {
    const cell = document.createElement("div");
    cell.className = "reference";
    const heading = document.createElement("h4");
    heading.textContent = ref.label;
    const link = document.createElement("a");
    link.href = `https://www.youtube.com/watch?v=${ref.video_id}&t=${Math.floor(ref.start)}s`;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = `Open on YouTube at ${clock(ref.start)} ↗`;
    cell.append(heading);
    if (ref.embeddable) {
      const frame = document.createElement("div");
      frame.className = "video-frame";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "video-load";
      button.textContent = `▶ Play ${clock(ref.start)}–${clock(ref.start + ref.seconds)}`;
      button.setAttribute("aria-label", `${title}: play ${ref.label} from ${clock(ref.start)}`);
      button.addEventListener("click", async () => {
        const YT = await loadYouTube();
        const host = document.createElement("div");
        button.replaceWith(host);
        const player = new YT.Player(host, {
          host: "https://www.youtube-nocookie.com",
          videoId: ref.video_id,
          playerVars: {
            start: Math.floor(ref.start), end: Math.ceil(ref.start + ref.seconds),
            rel: 0, playsinline: 1,
          },
          events: {
            // `start` only takes whole seconds; the window begins at a fractional time.
            onReady: event => { event.target.seekTo(ref.start, true); event.target.playVideo(); },
            onStateChange: event => { if (event.data === YT.PlayerState.PLAYING) pauseAll(player); },
          },
        });
        players.push(player);
      });
      frame.append(button);
      cell.append(frame);
    } else {
      const note = document.createElement("p");
      note.className = "fine-print";
      note.textContent = "The uploader does not allow embedding.";
      cell.append(note);
    }
    cell.append(link);
    if (ref.alignment === "automatic") {
      const note = document.createElement("p");
      note.className = "fine-print alignment-note";
      note.textContent = "Starts where the piano window starts; not yet hand-aligned, so the song may lead or lag.";
      cell.append(note);
    }
    return cell;
  }

  const GROUPS = {
    karaokeys: {
      title: "KaraoKeysPH covers",
      note: "Every KaraoKeysPH song in the 476-song evaluation panel, hand-picked samples first.",
    },
    sing2piano: {
      title: "Sing2Piano covers",
      note: "Sing2Piano does not allow its videos to be embedded, so these covers open on YouTube.",
    },
  };

  function render(root, samples, order, showComments, grouped = true) {
    let group = null;
    let count = 0;
    for (const sample of samples) {
      if (grouped && sample.group !== group) {
        group = sample.group;
        const heading = document.createElement("h3");
        heading.className = "group-title";
        const size = samples.filter(other => other.group === group).length;
        heading.textContent = `${GROUPS[group].title} (${size})`;
        const note = document.createElement("p");
        note.className = "fine-print";
        note.textContent = GROUPS[group].note;
        root.append(heading, note);
      }
      count += 1;
      const article = document.createElement("article");
      article.className = "sample";
      const title = document.createElement("h3");
      title.textContent = `Sample ${String(count).padStart(3, "0")}`;
      const identifier = document.createElement("p");
      identifier.className = "sample-id";
      identifier.textContent = `${sample.item_id} · Window ${sample.window}`;
      article.append(title, identifier);

      const refs = document.createElement("div");
      refs.className = "references";
      for (const key of ["piano", "song"]) {
        refs.append(reference(sample.references[key], title.textContent));
      }
      article.append(refs);

      const tracks = document.createElement("div");
      tracks.className = "tracks";
      for (const key of order) {
        const track = sample.tracks[key];
        const cell = document.createElement("div");
        cell.className = "track";
        const name = document.createElement("h4");
        name.textContent = track.label;
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "none";
        audio.src = `${track.src}?v=loudness-matched-4`;
        audio.setAttribute("aria-label", `${title.textContent}: ${track.label}`);
        audio.addEventListener("play", () => pauseAll(audio));
        cell.append(name, audio);
        tracks.append(cell);
      }
      article.append(tracks);
      if (showComments && sample.comment) {
        const details = document.createElement("details");
        details.className = "listening-comment";
        const summary = document.createElement("summary");
        summary.textContent = "Authors’ listening comment";
        const comment = document.createElement("p");
        comment.textContent = sample.comment;
        details.append(summary, comment);
        article.append(details);
      }
      root.append(article);
    }
  }
  const samples = window.LISTENING_SAMPLES.samples;
  const byKey = new Map(samples.map(sample => [sample.key, sample]));
  const chosen = window.LISTENING_SAMPLES.baseline_samples.map(key => byKey.get(key));
  render(document.getElementById("highlight-samples"), chosen, ["pico", "muse", "ace"], false, false);
  render(document.getElementById("comparison-samples"), samples,
    ["base", "var", "rule", "pico"], true);

  function makeTable(root, headers, rows) {
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const value of headers) {
      const cell = document.createElement("th");
      cell.textContent = value;
      headRow.append(cell);
    }
    head.append(headRow);
    const body = document.createElement("tbody");
    for (const values of rows) {
      const row = document.createElement("tr");
      for (const value of values) {
        const cell = document.createElement("td");
        if (typeof value === "object") {
          cell.textContent = value.text;
          if (value.significant) cell.className = "significant";
        } else cell.textContent = value;
        row.append(cell);
      }
      body.append(row);
    }
    root.replaceChildren(head, body);
  }
  const fmtRatio = value => value.toFixed(2);
  makeTable(document.getElementById("system-mixtures"),
    ["System", "Baseline", "Function-selective", "PiCoGen", "Style draw"],
    window.DETAILED_DATA.systems.map(item => [item.system, fmtRatio(item.baseline),
      fmtRatio(item.functional), fmtRatio(item.picogen),
      Object.entries(item.styles).filter(([, value]) => value > 0)
        .map(([name, value]) => `${name} ${fmtRatio(value)}`).join(" · ")]));
  const fmtEstimate = value => value
    ? `${value.point.toFixed(4)} ± ${value.bootstrap_sd.toFixed(4)}` : "—";
  makeTable(document.getElementById("estimate-table"),
    ["System", "Spectral RMSE ↓", "APA ↑", "FAD ↓", "Bass + drums share", "Piano share"],
    window.DETAILED_DATA.estimates.map(item => [item.system, fmtEstimate(item.spectral_rmse),
      fmtEstimate(item.apa), fmtEstimate(item.fad), fmtEstimate(item.bass_drums_share),
      fmtEstimate(item.piano_share)]));
  const comparisonCell = value => ({
    text: `${value.point >= 0 ? "+" : ""}${value.point.toFixed(4)} ` +
      `[${value.ci95[0] >= 0 ? "+" : ""}${value.ci95[0].toFixed(4)}, ` +
      `${value.ci95[1] >= 0 ? "+" : ""}${value.ci95[1].toFixed(4)}]` +
      (value.excludes_zero ? "†" : ""),
    significant: value.excludes_zero
  });
  for (const [cohort, id] of [["real_piano", "real-piano-comparisons"],
                              ["muscriptor", "muscriptor-comparisons"]]) {
    makeTable(document.getElementById(id), ["Difference", "Spectral RMSE Δ", "APA Δ", "FAD Δ"],
      window.DETAILED_DATA.comparisons[cohort].map(item => [item.comparison,
        comparisonCell(item.spectral_rmse), comparisonCell(item.apa), comparisonCell(item.fad)]));
  }

  function selectCollection() {
    const sections = ["samples", "system-comparison", "detailed"];
    // `#highlights` was this tab's earlier name; old links still land on it.
    const hash = window.location.hash.slice(1) === "highlights" ? "samples" : window.location.hash.slice(1);
    const selected = sections.includes(hash) ? hash : "samples";
    for (const section of sections) {
      document.getElementById(section).hidden = section !== selected;
    }
    for (const [id, active] of [["samples-tab", selected === "samples"],
      ["comparison-tab", selected === "system-comparison"],
      ["detailed-tab", selected === "detailed"]]) {
      const tab = document.getElementById(id);
      if (active) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    }
    pauseAll(null);
  }
  window.addEventListener("hashchange", selectCollection);
  selectCollection();
})();
