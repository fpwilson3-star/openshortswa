import os
import shutil
import subprocess


CAPTION_FONT_DIR = "fonts"
CAPTION_FONT_NAME = "Coolvetica"
CAPTION_FONT_PATH = os.path.join(CAPTION_FONT_DIR, "Coolvetica.ttf")


def ensure_caption_font():
    """Caption font is a licensed local font bundled in fonts/, not auto-downloaded."""
    if not os.path.exists(CAPTION_FONT_PATH):
        print(f"⚠️  Caption font missing at {CAPTION_FONT_PATH} — libass will fall back to its default")
    return CAPTION_FONT_PATH


def transcribe_audio(video_path):
    """
    Transcribe audio from a video file using faster-whisper.
    Returns transcript in the same format as main.py for compatibility.
    """
    from faster_whisper import WhisperModel

    print(f"🎙️  Transcribing audio from: {video_path}")

    # Run on CPU with INT8 quantization for speed
    model = WhisperModel("base", device="cpu", compute_type="int8")

    segments, info = model.transcribe(video_path, word_timestamps=True)

    transcript = {
        "segments": [],
        "language": info.language
    }

    for segment in segments:
        seg_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": []
        }
        if segment.words:
            for word in segment.words:
                seg_data["words"].append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end
                })
        transcript["segments"].append(seg_data)

    print(f"✅ Transcription complete. Language: {info.language}")
    return transcript


def generate_srt_from_video(video_path, output_path, max_chars=20, max_duration=2.0):
    """
    Transcribe a video and generate SRT directly.
    Used for dubbed videos that don't have a pre-existing transcript.
    """
    transcript = transcribe_audio(video_path)

    # Get video duration to use as clip_end
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0
    cap.release()

    return generate_srt(transcript, 0, duration, output_path, max_chars, max_duration)


def generate_srt(transcript, clip_start, clip_end, output_path, max_chars=20, max_duration=2.0):
    """
    Generates an SRT file from the transcript for a specific time range.
    Groups words into short lines suitable for vertical video.
    """
    
    words = []
    # 1. Extract and flatten words within range
    for segment in transcript.get('segments', []):
        for word_info in segment.get('words', []):
            # Check overlap
            if word_info['end'] > clip_start and word_info['start'] < clip_end:
                words.append(word_info)
    
    if not words:
        return False

    srt_content = ""
    index = 1
    
    current_block = []
    block_start = None
    
    for i, word in enumerate(words):
        # Adjust times relative to clip
        start = max(0, word['start'] - clip_start)
        end = max(0, word['end'] - clip_start)
        
        # Clip to video duration logic handled by ffmpeg usually, but good to be safe
        
        if not current_block:
            current_block.append(word)
            block_start = start
        else:
            # Decide whether to close block
            current_text_len = sum(len(w['word']) + 1 for w in current_block)
            duration = end - block_start
            
            if current_text_len + len(word['word']) > max_chars or duration > max_duration:
                # Finalize current block
                # End time of block is start of this word (gap) or end of last word?
                # Usually end of last word.
                block_end = current_block[-1]['end'] - clip_start
                
                text = " ".join([w['word'] for w in current_block]).strip()
                srt_content += format_srt_block(index, block_start, block_end, text)
                index += 1
                
                current_block = [word]
                block_start = start
            else:
                current_block.append(word)
    
    # Final block
    if current_block:
        block_end = current_block[-1]['end'] - clip_start
        text = " ".join([w['word'] for w in current_block]).strip()
        srt_content += format_srt_block(index, block_start, block_end, text)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)
        
    return True

def format_srt_block(index, start, end, text):
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
    return f"{index}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"

def hex_to_ass_color(hex_color, opacity=1.0):
    """Convert #RRGGBB to ASS &HAABBGGRR format. opacity: 0.0=transparent, 1.0=opaque"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        hex_color = "FFFFFF"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    alpha = round((1.0 - opacity) * 255)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def burn_subtitles(video_path, srt_path, output_path, alignment=2, fontsize=16,
                   font_name="Verdana", font_color="#FFFFFF",
                   border_color="#000000", border_width=2,
                   bg_color="#000000", bg_opacity=0.0):
    """
    Burns subtitles into the video using FFmpeg.
    Supports two modes:
    - Outline mode (bg_opacity=0): Text with colored outline/border
    - Box mode (bg_opacity>0): Text with semi-transparent background box
    """
    # Position mapping
    ass_alignment = 2
    align_lower = str(alignment).lower()
    if align_lower == 'top':
        ass_alignment = 6
    elif align_lower == 'middle':
        ass_alignment = 10
    elif align_lower == 'bottom':
        ass_alignment = 2

    # Font size scaling for ASS virtual resolution (PlayResY=288 default)
    # For vertical 1080x1920 video, we need larger text for readability
    final_fontsize = int(fontsize * 0.85)
    if final_fontsize < 10:
        final_fontsize = 10

    # Path handling for FFmpeg filter syntax
    safe_srt_path = srt_path.replace('\\', '/').replace(':', '\\:')

    # Convert colors to ASS format and build style
    primary_colour = hex_to_ass_color(font_color, 1.0)

    if bg_opacity > 0:
        # Box mode: opaque background box
        border_style = 3
        outline_colour = hex_to_ass_color(bg_color, bg_opacity)
        outline_width = 1
    else:
        # Outline mode: text border/outline
        border_style = 1
        outline_colour = hex_to_ass_color(border_color, 1.0)
        outline_width = max(1, border_width)

    back_colour = hex_to_ass_color("#000000", 0.0)

    style_string = (
        f"Alignment={ass_alignment},"
        f"Fontname={font_name},"
        f"Fontsize={final_fontsize},"
        f"PrimaryColour={primary_colour},"
        f"OutlineColour={outline_colour},"
        f"BackColour={back_colour},"
        f"BorderStyle={border_style},"
        f"Outline={outline_width},"
        f"Shadow=0,"
        f"MarginV=25,"
        f"Bold=1"
    )

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f"subtitles='{safe_srt_path}':force_style='{style_string}'",
        '-c:a', 'copy',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        output_path
    ]

    print(f"🎬 Burning subtitles: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"❌ FFmpeg Subtitle Error: {result.stderr.decode()}")
        raise Exception(f"FFmpeg failed: {result.stderr.decode()}")

    return True


def _group_words_into_blocks(words, max_chars=22, max_duration=2.0):
    blocks = []
    current = []
    for w in words:
        if not current:
            current = [w]
            continue
        text_len = sum(len(x['word']) + 1 for x in current)
        duration = w['end'] - current[0]['start']
        if text_len + len(w['word']) > max_chars or duration > max_duration:
            blocks.append(current)
            current = [w]
        else:
            current.append(w)
    if current:
        blocks.append(current)
    return blocks


def _ass_time(t):
    if t < 0:
        t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    cs = int(round((s - int(s)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"


def generate_karaoke_ass(transcript, clip_start, clip_end, output_path,
                        play_res_x=1080, play_res_y=1920,
                        max_chars=22, max_duration=2.0,
                        font_name=CAPTION_FONT_NAME, fontsize=78,
                        normal_color="#FFFFFF", highlight_color="#FFEE00",
                        outline_color="#000000", outline_width=4,
                        margin_v=240):
    """
    Build a karaoke-style ASS file: one event per word, each showing the full
    block with that word highlighted. Words within a block stay visible during
    gaps by extending each event to the start of the next word.
    """
    words = []
    for segment in transcript.get('segments', []):
        for w in segment.get('words', []):
            if w['end'] > clip_start and w['start'] < clip_end:
                words.append(w)

    if not words:
        return False

    blocks = _group_words_into_blocks(words, max_chars=max_chars, max_duration=max_duration)

    primary = hex_to_ass_color(normal_color, 1.0)
    outline = hex_to_ass_color(outline_color, 1.0)
    highlight = hex_to_ass_color(highlight_color, 1.0)
    back = hex_to_ass_color("#000000", 0.0)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, "
        "Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{fontsize},{primary},{primary},{outline},{back},"
        f"-1,0,0,0,100,100,0,0,1,{outline_width},0,2,60,60,{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for block in blocks:
        block_end_time = block[-1]['end'] - clip_start
        for i, w in enumerate(block):
            start = w['start'] - clip_start
            end = block[i + 1]['start'] - clip_start if i + 1 < len(block) else block_end_time
            parts = []
            for j, ww in enumerate(block):
                clean = ww['word'].strip()
                if not clean:
                    continue
                if j == i:
                    parts.append(f"{{\\c{highlight}}}{clean}{{\\r}}")
                else:
                    parts.append(clean)
            text = " ".join(parts)
            lines.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}"
            )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    return True


def burn_karaoke_subtitles(video_path, ass_path, output_path, fonts_dir=CAPTION_FONT_DIR):
    """Burn karaoke ASS subtitles. fontsdir points libass at the bundled font.

    Copies the ASS file to a sanitized path before invoking ffmpeg so apostrophes
    and other shell-special chars in the original filename don't break the
    `-vf subtitles='...'` single-quoted filter syntax.
    """
    safe_dir = os.path.dirname(ass_path) or '.'
    safe_ass_path = os.path.join(safe_dir, '_caps.ass')
    shutil.copy(ass_path, safe_ass_path)

    safe_ass = safe_ass_path.replace('\\', '/').replace(':', '\\:')
    safe_fonts = fonts_dir.replace('\\', '/').replace(':', '\\:')

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f"subtitles='{safe_ass}':fontsdir='{safe_fonts}'",
        '-c:a', 'copy',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        output_path
    ]

    print(f"🎤 Burning karaoke subtitles: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    finally:
        if os.path.exists(safe_ass_path):
            os.remove(safe_ass_path)

    if result.returncode != 0:
        print(f"❌ FFmpeg Karaoke Error: {result.stderr.decode()}")
        raise Exception(f"FFmpeg failed: {result.stderr.decode()}")

    return True

