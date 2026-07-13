"""
broll.py — automated full-frame B-roll cutaways for talking-head clips.

MOCKUP / PROTOTYPE.

For a single finished clip, this:
  1. plan_broll_cues()  — Gemini reads the clip's transcript window and picks N
     cutaway moments (N derived from clip length: ~1 every 10 s). Each cue has a
     clip-relative start/end and a Pexels search query.
  2. fetch_pexels_asset() — downloads a portrait photo or video for each query.
  3. normalize_asset()  — renders each asset to a 1080x1920 H.264 segment of the
     EXACT cutaway length (cover-crop so it fills the frame; a slow Ken Burns
     zoom on stills so they aren't static).
  4. apply_broll()      — ONE FFmpeg pass overlays every segment onto the clip
     at its window via enable='between(t,a,b)'. The clip's audio is copied
     through untouched, so the speaker keeps talking under every cutaway.

ORDERING: run this BEFORE add_hook_to_video() and subtitle burning so the hook
text and karaoke captions render ON TOP of the cutaways (full-frame B-roll would
otherwise cover them). In main.py's per-clip loop that means: cut → vertical →
**B-roll** → hook → captions → outro.

Pexels is free (200 req/hr, 20k/mo by default; raised on request). Their terms
ask for a visible "Photos provided by Pexels" credit somewhere in the app — not
burned into the video. Set PEXELS_API_KEY in the environment.
"""

import argparse
import json
import os
import subprocess
import tempfile

import httpx

from gemini_utils import GEMINI_LITE_MODELS, gemini_generate_with_fallback

# --- Output format (matches the rest of the pipeline) ---
OUT_W, OUT_H = 1080, 1920
FPS = 30

# --- Cutaway pacing ---
SECONDS_PER_CUTAWAY = 10.0   # average one cutaway per this many seconds of clip
MIN_CUTAWAY = 2.0            # each cutaway lasts between MIN and MAX seconds
MAX_CUTAWAY = 4.0
LEAD_GUARD = 1.5            # don't cover the very first beat (the spoken hook)
TAIL_GUARD = 1.2            # leave the speaker's resolving beat at the end uncovered

PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"


# ---------------------------------------------------------------------------
# 1. Planning
# ---------------------------------------------------------------------------

def target_cutaway_count(duration):
    """How many cutaways for a clip of `duration` seconds (~1 per 10 s)."""
    return max(1, round(duration / SECONDS_PER_CUTAWAY))


_PLAN_PROMPT = """\
You are a short-form video editor adding B-roll cutaways to a talking-head clip
from "Wellness Actually", a credible health/medicine podcast.

The clip is {duration:.1f} seconds long. Insert EXACTLY {count} full-frame B-roll
cutaways. Each cutaway briefly replaces the talking head with an illustrative
image or stock video while the host keeps talking underneath.

Rules:
- Place each cutaway on a CONCRETE, VISUALIZABLE noun or moment in the transcript
  (an organ, a food, a lab test, an activity, a place, an object). NEVER place one
  on an abstract or unvisualizable statement.
- `query` is a literal Pexels stock-search phrase for that visual. Keep it simple
  and physical ("brain mri scan", "fresh broccoli", "person jogging at sunrise").
  AVOID queries that would return people talking to camera — we already have that.
- Spread the cutaways across the clip; do not overlap them.
- Each cutaway is {min_dur:.1f}-{max_dur:.1f} seconds long.
- Keep cutaways between {lead:.1f}s and {tail_limit:.1f}s (leave the open and the
  final beat on the host's face).
- `type` is "photo" (preferred — sharper, more reliable) or "video" (only when
  motion clearly helps, e.g. running, cooking, waves).

TRANSCRIPT (clip-relative seconds):
{words_json}

Return ONLY valid JSON, no markdown:
{{
  "cutaways": [
    {{"start": <sec>, "end": <sec>, "type": "photo|video",
      "query": "<pexels search phrase>", "reason": "<the word/idea it illustrates>"}}
  ]
}}
"""


def plan_broll_cues(words_rel, duration, client=None):
    """Ask Gemini for cutaway cues. `words_rel`: [{'w','s','e'}] clip-relative.

    Returns a sanitized, non-overlapping, in-bounds list of cue dicts.
    """
    count = target_cutaway_count(duration)
    if client is None:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = _PLAN_PROMPT.format(
        duration=duration, count=count,
        min_dur=MIN_CUTAWAY, max_dur=MAX_CUTAWAY,
        lead=LEAD_GUARD, tail_limit=max(LEAD_GUARD, duration - TAIL_GUARD),
        words_json=json.dumps(words_rel),
    )

    response, _ = gemini_generate_with_fallback(
        client, prompt, models=GEMINI_LITE_MODELS,
        config={"response_mime_type": "application/json"},
    )
    if response is None:
        print("⚠️  B-roll planning failed (Gemini). No cutaways.")
        return []

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    try:
        cues = json.loads(text).get("cutaways", [])
    except Exception as e:
        print(f"⚠️  Could not parse B-roll plan: {e}")
        return []

    return _sanitize_cues(cues, duration, count)


def _sanitize_cues(cues, duration, count):
    """Clamp to bounds, enforce min/max length, drop overlaps, cap at `count`."""
    lo, hi = LEAD_GUARD, duration - TAIL_GUARD
    clean = []
    for c in cues:
        try:
            s = float(c["start"]); e = float(c["end"])
        except (KeyError, ValueError, TypeError):
            continue
        s = max(lo, s)
        e = min(hi, max(s + MIN_CUTAWAY, e))
        e = min(e, s + MAX_CUTAWAY)
        if e - s < MIN_CUTAWAY or s >= hi:
            continue
        query = (c.get("query") or "").strip()
        if not query:
            continue
        clean.append({
            "start": round(s, 3), "end": round(e, 3),
            "type": "video" if c.get("type") == "video" else "photo",
            "query": query, "reason": c.get("reason", ""),
        })

    clean.sort(key=lambda c: c["start"])
    spaced, last_end = [], -1.0
    for c in clean:
        if c["start"] < last_end:          # overlaps the previous cutaway
            continue
        spaced.append(c)
        last_end = c["end"]
        if len(spaced) >= count:
            break
    return spaced


# ---------------------------------------------------------------------------
# 2. Pexels fetch
# ---------------------------------------------------------------------------

def fetch_pexels_asset(query, kind, api_key, dest_dir):
    """Download the best portrait photo/video for `query`. Returns a file path
    (with a `.is_video` style suffix) or None if nothing usable was found."""
    headers = {"Authorization": api_key}
    is_video = kind == "video"
    url = PEXELS_VIDEO_URL if is_video else PEXELS_PHOTO_URL
    params = {"query": query, "orientation": "portrait", "per_page": 8}

    try:
        r = httpx.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"   ⚠️  Pexels search failed for '{query}': {e}")
        return None

    if is_video:
        videos = data.get("videos", [])
        if not videos:
            return None
        # Prefer the smallest video file at least 1920 tall; else the tallest.
        files = sorted(videos[0].get("video_files", []),
                       key=lambda f: (f.get("height") or 0))
        tall_enough = [f for f in files if (f.get("height") or 0) >= OUT_H]
        pick = (tall_enough[0] if tall_enough else (files[-1] if files else None))
        link = pick.get("link") if pick else None
        ext = ".mp4"
    else:
        photos = data.get("photos", [])
        if not photos:
            return None
        src = photos[0].get("src", {})
        link = src.get("original") or src.get("large2x") or src.get("portrait")
        ext = ".jpg"

    if not link:
        return None

    try:
        path = os.path.join(dest_dir, f"broll_{abs(hash(query)) % 10**8}{ext}")
        with httpx.stream("GET", link, timeout=60, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_bytes(1 << 16):
                    f.write(chunk)
        return path
    except Exception as e:
        print(f"   ⚠️  Pexels download failed for '{query}': {e}")
        return None


# ---------------------------------------------------------------------------
# 3. Normalize each asset to a 1080x1920 segment of the exact cutaway length
# ---------------------------------------------------------------------------

def _run_ffmpeg(cmd, what):
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg {what} failed: {res.stderr.decode()[:400]}")


# Cover-crop to fill 9:16, regardless of source aspect.
_COVER = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
          f"crop={OUT_W}:{OUT_H}")


def normalize_asset(asset_path, is_video, duration, out_path):
    """Render `asset_path` to a 1080x1920, FPS, silent H.264 clip of `duration`s."""
    frames = max(1, int(round(duration * FPS)))

    if is_video:
        vf = f"{_COVER},fps={FPS},format=yuv420p"
        cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", asset_path,
               "-t", f"{duration:.3f}", "-an", "-vf", vf,
               "-r", str(FPS), "-c:v", "libx264", "-preset", "fast",
               "-crf", "20", out_path]
    else:
        # Ken Burns: a slow zoom-in on a still. zoompan crops its INPUT at integer
        # pixel positions every frame, so on a small source the rounding makes the
        # pan shimmer by a pixel (the "janky" stutter). Fix: supersample the still
        # to a large intermediate first (SS× the frame) so each 1-px rounding step
        # is an invisible fraction of a percent; zoompan then downsamples to
        # 1080x1920, which also anti-aliases the motion.
        SS = 4
        big_w, big_h = OUT_W * SS, OUT_H * SS
        zoom = (f"scale={big_w}:{big_h}:"
                f"force_original_aspect_ratio=increase,"
                f"crop={big_w}:{big_h},"
                f"zoompan=z='min(zoom+0.0008,1.15)'"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d={frames}:s={OUT_W}x{OUT_H}:fps={FPS},format=yuv420p")
        cmd = ["ffmpeg", "-y", "-i", asset_path, "-vf", zoom,
               "-frames:v", str(frames), "-r", str(FPS),
               "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", out_path]

    _run_ffmpeg(cmd, "normalize")
    return out_path


# ---------------------------------------------------------------------------
# 4. Overlay every segment onto the clip (single pass, audio untouched)
# ---------------------------------------------------------------------------

def apply_broll(video_path, segments, output_path):
    """`segments`: [(seg_path, start, end)]. Full-frame overlay of each segment
    at its window. The clip's own audio is copied through unchanged."""
    if not segments:
        return False

    inputs = ["-i", video_path]
    for seg_path, _, _ in segments:
        inputs += ["-i", seg_path]

    # Shift each segment's PTS to its window start so overlay frames line up,
    # then chain opaque full-frame overlays gated by enable=between(t,a,b).
    parts, last = [], "[0:v]"
    for idx, (_, start, end) in enumerate(segments, start=1):
        parts.append(f"[{idx}:v]setpts=PTS-STARTPTS+{start:.3f}/TB[s{idx}];")
        out = f"[o{idx}]"
        parts.append(f"{last}[s{idx}]overlay=0:0:"
                     f"enable='between(t,{start:.3f},{end:.3f})'{out};")
        last = out
    filtergraph = "".join(parts).rstrip(";")

    cmd = ["ffmpeg", "-y", *inputs,
           "-filter_complex", filtergraph,
           "-map", last, "-map", "0:a?",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-c:a", "copy", output_path]
    _run_ffmpeg(cmd, "overlay")
    return True


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _clip_relative_words(transcript, clip_start, clip_end):
    """Words inside [clip_start, clip_end], rebased so the clip begins at 0."""
    words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            if w["start"] >= clip_start and w["end"] <= clip_end:
                words.append({"w": w["word"],
                              "s": round(w["start"] - clip_start, 3),
                              "e": round(w["end"] - clip_start, 3)})
    return words


def add_broll_to_video(video_path, transcript, clip_start, clip_end,
                       output_path, gemini_key=None, pexels_key=None):
    """Plan + fetch + insert B-roll cutaways into one finished clip.

    Returns True if cutaways were applied, False if it left the clip untouched
    (no key, no plan, or nothing fetched) — caller should keep the original.
    """
    pexels_key = pexels_key or os.getenv("PEXELS_API_KEY")
    if not pexels_key:
        print("   ⚠️  PEXELS_API_KEY not set — skipping B-roll.")
        return False

    duration = clip_end - clip_start
    words_rel = _clip_relative_words(transcript, clip_start, clip_end)
    if not words_rel:
        print("   ⚠️  No transcript words in clip range — skipping B-roll.")
        return False

    from google import genai
    client = genai.Client(api_key=gemini_key or os.environ["GEMINI_API_KEY"])
    cues = plan_broll_cues(words_rel, duration, client=client)
    if not cues:
        return False
    print(f"   🎞️  Planned {len(cues)} cutaway(s) "
          f"(~1 per {SECONDS_PER_CUTAWAY:.0f}s over {duration:.0f}s)")

    with tempfile.TemporaryDirectory(prefix="broll_") as tmp:
        segments = []
        for i, cue in enumerate(cues):
            print(f"      • {cue['start']:.1f}-{cue['end']:.1f}s "
                  f"[{cue['type']}] '{cue['query']}' — {cue['reason']}")
            asset = fetch_pexels_asset(cue["query"], cue["type"], pexels_key, tmp)
            if not asset:
                print("        ↳ no asset found, skipping this cutaway")
                continue
            seg = os.path.join(tmp, f"seg_{i}.mp4")
            try:
                normalize_asset(asset, cue["type"] == "video",
                                cue["end"] - cue["start"], seg)
            except Exception as e:
                print(f"        ↳ normalize failed: {e}")
                continue
            segments.append((seg, cue["start"], cue["end"]))

        if not segments:
            print("   ⚠️  No B-roll assets usable — leaving clip unchanged.")
            return False

        return apply_broll(video_path, segments, output_path)


# ---------------------------------------------------------------------------
# CLI — test on one already-cut clip without running the whole pipeline
# ---------------------------------------------------------------------------

def _main():
    from dotenv import load_dotenv
    load_dotenv()
    p = argparse.ArgumentParser(description="Mock up B-roll cutaways on a clip.")
    p.add_argument("--input", required=True, help="A finished 1080x1920 clip.")
    p.add_argument("--output", required=True, help="Where to write the result.")
    p.add_argument("--metadata", help="*_metadata.json from main.py (has transcript + shorts).")
    p.add_argument("--clip-index", type=int, default=0,
                   help="Which entry in metadata['shorts'] this clip is (0-based).")
    p.add_argument("--queries",
                   help="Bypass Gemini: ';'-separated literal Pexels queries, "
                        "spaced evenly across the clip (for testing the FFmpeg path).")
    args = p.parse_args()

    if args.queries:
        # Probe the clip duration and lay queries out evenly — no Gemini needed.
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", args.input]).decode().strip())
        qs = [q.strip() for q in args.queries.split(";") if q.strip()]
        usable = dur - LEAD_GUARD - TAIL_GUARD
        slot = usable / len(qs)
        cues = []
        for i, q in enumerate(qs):
            s = LEAD_GUARD + i * slot + max(0, (slot - MAX_CUTAWAY) / 2)
            cues.append({"start": round(s, 3),
                         "end": round(min(s + MAX_CUTAWAY, dur - TAIL_GUARD), 3),
                         "type": "photo", "query": q, "reason": "(manual)"})
        pexels_key = os.environ["PEXELS_API_KEY"]
        with tempfile.TemporaryDirectory(prefix="broll_") as tmp:
            segs = []
            for i, c in enumerate(cues):
                print(f"   • {c['start']:.1f}-{c['end']:.1f}s '{c['query']}'")
                a = fetch_pexels_asset(c["query"], "photo", pexels_key, tmp)
                if not a:
                    continue
                seg = os.path.join(tmp, f"seg_{i}.mp4")
                normalize_asset(a, False, c["end"] - c["start"], seg)
                segs.append((seg, c["start"], c["end"]))
            ok = apply_broll(args.input, segs, args.output)
        print("✅ Done." if ok else "⚠️  Nothing applied.")
        return

    if not args.metadata:
        p.error("provide --metadata (for transcript + clip range) or --queries")

    with open(args.metadata) as f:
        meta = json.load(f)
    clip = meta["shorts"][args.clip_index]
    ok = add_broll_to_video(args.input, meta["transcript"],
                            clip["start"], clip["end"], args.output)
    print("✅ Done." if ok else "⚠️  Nothing applied (clip left unchanged).")


if __name__ == "__main__":
    _main()
