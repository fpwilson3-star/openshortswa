import os
import re
import textwrap
import subprocess
from PIL import Image, ImageDraw, ImageFont, ImageFilter

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U0001FB00-\U0001FBFF"
    "\U00002600-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "‍"
    "️"
    "]+",
    flags=re.UNICODE,
)

def _strip_emojis(text):
    return _EMOJI_RE.sub("", text or "").strip()

FONT_DIR = "fonts"
FONT_PATH = os.path.join(FONT_DIR, "Anton.ttf")
FALLBACK_FONT_PATH = os.path.join(FONT_DIR, "Coolvetica.ttf")

# Matches the karaoke caption highlight (#FFEE00) so each clip has one accent color.
HIGHLIGHT_COLOR = (255, 238, 0, 255)
TEXT_COLOR = (255, 255, 255, 255)

_EMPHASIS_RE = re.compile(r"\*([^*]+)\*")

def _parse_emphasis(text):
    """
    Splits text into (word, highlighted) tokens. Words wrapped in *asterisks*
    are flagged for highlight rendering; stray asterisks are stripped.
    """
    tokens = []
    pos = 0
    for m in _EMPHASIS_RE.finditer(text):
        for w in text[pos:m.start()].split():
            tokens.append((w, False))
        for w in m.group(1).split():
            tokens.append((w, True))
        pos = m.end()
    for w in text[pos:].split():
        tokens.append((w, False))
    return [(w.replace('*', ''), hl) for w, hl in tokens if w.replace('*', '')]

def download_font_if_needed():
    """Hook font is bundled in fonts/, not auto-downloaded."""
    if not os.path.exists(FONT_PATH):
        print(f"⚠️  Hook font missing at {FONT_PATH} — falling back to {FALLBACK_FONT_PATH}")
    return os.path.exists(FONT_PATH)

def create_hook_image(text, target_width, output_image_path="hook_overlay.png", font_scale=1.0):
    """
    Renders all-caps stroke text directly (no background card): white Anton
    with a thick black outline and a soft drop shadow, *emphasized* words in
    highlight yellow. target_width: max width the text may occupy.
    """
    download_font_if_needed()

    # ~80px at 1080-wide video — reads ~1.4x the karaoke captions (Anton runs tall).
    base_font_size = int(target_width * 0.082)
    font_size = int(base_font_size * font_scale)
    stroke_width = max(3, round(font_size * 0.085))
    # Margin must clear the stroke plus the blurred shadow on every side.
    margin = stroke_width + 14
    shadow_offset = (0, int(font_size * 0.07))

    font = None
    for path in (FONT_PATH, FALLBACK_FONT_PATH):
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception as e:
            print(f"⚠️ Could not load font {path}: {e}")
    if font is None:
        font = ImageFont.load_default()

    ascent, descent = font.getmetrics()
    visual_line_height = ascent + descent
    extra_line_gap = int(visual_line_height * 0.08)
    line_advance = visual_line_height + extra_line_gap

    tokens = _parse_emphasis(text or "")
    tokens = [(w.upper(), hl) for w, hl in tokens]

    # Pixel-based wrap into lines of tokens.
    max_text_width = target_width - (2 * margin)
    lines = []
    current = []
    for token in tokens:
        candidate = ' '.join([t[0] for t in current] + [token[0]])
        if font.getlength(candidate) <= max_text_width or not current:
            current.append(token)
        else:
            lines.append(current)
            current = [token]
    if current:
        lines.append(current)
    if not lines:
        lines = [[]]

    space_w = font.getlength(' ')
    line_widths = [font.getlength(' '.join(t[0] for t in line)) for line in lines]
    max_line_width = int(max(line_widths, default=0))

    canvas_w = max_line_width + (2 * margin)
    total_text_height = len(lines) * visual_line_height + max(0, len(lines) - 1) * extra_line_gap
    canvas_h = total_text_height + (2 * margin)

    def draw_text_layer(draw, dx, dy, fill, stroke_fill, highlight_fill):
        baseline_y = margin + ascent + dy
        for line, line_w in zip(lines, line_widths):
            x = (canvas_w - int(line_w)) // 2 + dx
            for word, highlighted in line:
                draw.text(
                    (x, baseline_y), word, font=font, anchor="ls",
                    fill=highlight_fill if highlighted else fill,
                    stroke_width=stroke_width, stroke_fill=stroke_fill,
                )
                x += font.getlength(word) + space_w
            baseline_y += line_advance

    # Soft drop shadow: blurred black copy of the text, offset downward.
    shadow = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw_text_layer(ImageDraw.Draw(shadow), shadow_offset[0], shadow_offset[1],
                    (0, 0, 0, 170), (0, 0, 0, 170), (0, 0, 0, 170))
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))

    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    img = Image.alpha_composite(img, shadow)
    draw_text_layer(ImageDraw.Draw(img), 0, 0, TEXT_COLOR, (0, 0, 0, 255), HIGHLIGHT_COLOR)

    img.save(output_image_path)
    return output_image_path, canvas_w, canvas_h

def add_hook_to_video(video_path, text, output_path, position="top", font_scale=1.0, duration=None):
    """
    Overlays text hook onto video.
    position: 'top', 'center', 'bottom'
    font_scale: float multiplier (1.0 = default)
    duration: if set, overlay only displays for the first `duration` seconds
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video {video_path} not found")

    text = _strip_emojis(text)
    if not text:
        raise ValueError("Hook text is empty after stripping emojis")

    # 1. Probe video width to scale text properly
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_path]
        res = subprocess.check_output(cmd).decode().strip()
        # Takes first stream if multiple
        dims = res.split('\n')[0].split('x')
        video_width = int(dims[0])
        video_height = int(dims[1])
    except Exception as e:
        print(f"⚠️ FFprobe failed: {e}. Assuming 1080x1920")
        video_width = 1080
        video_height = 1920

    # 2. Generate Image
    # Box check: Don't let it be wider than 90% of screen
    target_box_width = int(video_width * 0.9)

    hook_filename = f"temp_hook_{os.path.basename(video_path)}.png"
    # Ensure unique or temp location if needed, but relative is fine for this app structure

    try:
        img_path, box_w, box_h = create_hook_image(text, target_box_width, hook_filename, font_scale=font_scale)

        # 3. Calculate Overlay Position
        overlay_x = (video_width - box_w) // 2

        if position == "center":
            overlay_y = (video_height - box_h) // 2
        elif position == "bottom":
             # Bottom 20% mark (approx)
             overlay_y = int(video_height * 0.70)
        else:
             # Top ~8% mark — near the top edge but clear of phone notch/status overlays
             overlay_y = int(video_height * 0.08)

        # 4. FFmpeg Command
        print(f"🎬 Overlaying hook: '{text}' at {overlay_x},{overlay_y}")

        overlay_expr = f"[0:v][1:v]overlay={overlay_x}:{overlay_y}"
        if duration is not None:
            overlay_expr += f":enable='between(t,0,{duration})'"

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', img_path,
            '-filter_complex', overlay_expr,
            '-c:a', 'copy',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            output_path
        ]

        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Hook added to {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error: {e.stderr.decode() if e.stderr else 'Unknown'}")
        raise e
    except Exception as e:
        print(f"❌ Hook Gen Error: {e}")
        raise e
    finally:
        # Cleanup temp image
        if os.path.exists(hook_filename):
            os.remove(hook_filename)
