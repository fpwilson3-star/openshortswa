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
FONT_PATH = os.path.join(FONT_DIR, "Coolvetica.ttf")

def download_font_if_needed():
    """Hook font is a licensed local font — bundled in fonts/, not auto-downloaded."""
    if not os.path.exists(FONT_PATH):
        print(f"⚠️  Hook font missing at {FONT_PATH} — falling back to PIL default")
    return os.path.exists(FONT_PATH)

def create_hook_image(text, target_width, output_image_path="hook_overlay.png", font_scale=1.0):
    """
    Generates a white rounded box with bold sans-serif text, vertically and
    horizontally centered using font.getmetrics() for consistent line heights.
    target_width: max width the box should occupy (e.g. 85% of video).
    """
    download_font_if_needed()

    padding_x = 36
    padding_y = 30
    cornerradius = 20
    shadow_offset = (5, 5)

    # Inter Bold is a bit denser than Noto Serif Bold — slightly smaller ratio.
    base_font_size = int(target_width * 0.046)
    font_size = int(base_font_size * font_scale)

    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception as e:
        print(f"⚠️ Could not load font {FONT_PATH}, using default. Error: {e}")
        font = ImageFont.load_default()

    ascent, descent = font.getmetrics()
    visual_line_height = ascent + descent
    extra_line_gap = int(visual_line_height * 0.15)
    line_advance = visual_line_height + extra_line_gap

    # Text wrap (pixel-based using font.getlength for speed)
    max_text_width = target_width - (2 * padding_x)
    paragraphs = (text or "").split('\n')
    lines = []
    for p in paragraphs:
        if not p.strip():
            lines.append("")
            continue
        words = p.split()
        current = []
        for word in words:
            candidate = ' '.join(current + [word])
            if font.getlength(candidate) <= max_text_width:
                current.append(word)
            else:
                if current:
                    lines.append(' '.join(current))
                    current = [word]
                else:
                    lines.append(word)
                    current = []
        if current:
            lines.append(' '.join(current))

    if not lines:
        lines = [""]

    max_line_width = max((font.getlength(l) for l in lines), default=0)
    box_width = max(int(max_line_width) + (2 * padding_x), int(target_width * 0.3))

    total_text_height = len(lines) * visual_line_height + max(0, len(lines) - 1) * extra_line_gap
    box_height = total_text_height + (2 * padding_y)

    canvas_w = box_width + 40
    canvas_h = box_height + 40

    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    shadow_box = [
        (20 + shadow_offset[0], 20 + shadow_offset[1]),
        (20 + box_width + shadow_offset[0], 20 + box_height + shadow_offset[1]),
    ]
    draw.rounded_rectangle(shadow_box, radius=cornerradius, fill=(0, 0, 0, 100))
    img = img.filter(ImageFilter.GaussianBlur(5))

    draw_final = ImageDraw.Draw(img)
    main_box = [(20, 20), (20 + box_width, 20 + box_height)]
    draw_final.rounded_rectangle(main_box, radius=cornerradius, fill=(255, 255, 255, 240))

    # Draw lines positioned by baseline — consistent vertical rhythm regardless of glyph content.
    baseline_y = 20 + padding_y + ascent
    for line in lines:
        if line:
            line_w = font.getlength(line)
            x = 20 + (box_width - int(line_w)) // 2
            draw_final.text((x, baseline_y), line, font=font, fill="black", anchor="ls")
        baseline_y += line_advance

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
