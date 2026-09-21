from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

FULL_TEXT = "Chagatai Sentence Segmentation"
HIGHLIGHT_WORD = "Chagatai"
REST_WORD = " Sentence Segmentation"

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_SIZE = 58
CANVAS_WIDTH = 1150
CANVAS_HEIGHT = 100


def create_typing_gif(
    output_path: Path,
    *,
    highlight_color: tuple[int, int, int, int],
    text_color: tuple[int, int, int, int],
    cursor_color: tuple[int, int, int, int],
) -> None:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # Compute full text width to center the final block
    bbox_full = font.getbbox(FULL_TEXT)
    full_width = bbox_full[2] - bbox_full[0]
    start_x = (CANVAS_WIDTH - full_width) // 2
    baseline_y = (CANVAS_HEIGHT - (bbox_full[3] - bbox_full[1])) // 2

    # Measure highlight word width
    bbox_chg = font.getbbox(HIGHLIGHT_WORD)
    chg_width = bbox_chg[2] - bbox_chg[0]

    frames: list[Image.Image] = []
    durations: list[int] = []

    def render_frame(current_text: str, show_cursor: bool) -> Image.Image:
        # RGBA image for crisp text and transparent background
        img = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw text up to current_text
        if len(current_text) <= len(HIGHLIGHT_WORD):
            # Only in highlight word
            draw.text((start_x, baseline_y), current_text, font=font, fill=highlight_color)
            curr_bbox = font.getbbox(current_text) if current_text else (0, 0, 0, 0)
            cursor_x = start_x + (curr_bbox[2] - curr_bbox[0]) + 4
        else:
            # Highlight word in highlight_color, rest in text_color
            draw.text((start_x, baseline_y), HIGHLIGHT_WORD, font=font, fill=highlight_color)
            rest_part = current_text[len(HIGHLIGHT_WORD):]
            rest_x = start_x + chg_width
            draw.text((rest_x, baseline_y), rest_part, font=font, fill=text_color)
            rest_bbox = font.getbbox(rest_part) if rest_part else (0, 0, 0, 0)
            cursor_x = rest_x + (rest_bbox[2] - rest_bbox[0]) + 4

        if show_cursor:
            cursor_h = FONT_SIZE - 4
            cursor_y = baseline_y + 2
            draw.rectangle(
                [cursor_x, cursor_y, cursor_x + 4, cursor_y + cursor_h],
                fill=cursor_color,
            )

        return img

    # 1. Initial pause with blinking cursor
    frames.append(render_frame("", show_cursor=True))
    durations.append(400)
    frames.append(render_frame("", show_cursor=False))
    durations.append(250)

    # 2. Typing out letters
    for i in range(1, len(FULL_TEXT) + 1):
        sub = FULL_TEXT[:i]
        frames.append(render_frame(sub, show_cursor=True))
        # Variable natural typing speed
        if FULL_TEXT[i - 1] == " ":
            durations.append(110)
        elif i == len(HIGHLIGHT_WORD):
            durations.append(140)
        else:
            durations.append(70)

    # 3. Blinking cursor pause at the end
    for _ in range(3):
        frames.append(render_frame(FULL_TEXT, show_cursor=True))
        durations.append(450)
        frames.append(render_frame(FULL_TEXT, show_cursor=False))
        durations.append(400)

    # Convert RGBA frames to palette mode preserving alpha
    # For GIF transparency with crisp edges, quantize each frame
    paletted_frames: list[Image.Image] = []
    for frame in frames:
        # Use alpha mask for transparent background
        alpha = frame.split()[3]
        # Create RGB image with white/black key for quantization
        rgb_frame = frame.convert("RGB")
        # Quantize to 255 colors, reserving index 0 for transparency
        p_frame = rgb_frame.convert("P", palette=Image.ADAPTIVE, colors=254)
        # Set pixels where alpha < 128 to transparent index
        mask = Image.eval(alpha, lambda a: 255 if a < 128 else 0)
        p_frame.paste(255, mask)
        p_frame.info["transparency"] = 255
        paletted_frames.append(p_frame)

    paletted_frames[0].save(
        output_path,
        save_all=True,
        append_images=paletted_frames[1:],
        duration=durations,
        loop=0,
        transparency=255,
        disposal=2,
    )
    print(f"Saved: {output_path} ({len(frames)} frames)")


def main() -> None:
    # Light theme: dark slate text, emerald highlight
    create_typing_gif(
        ASSETS_DIR / "title_light.gif",
        highlight_color=(5, 150, 105, 255),    # #059669 emerald-600
        text_color=(31, 35, 40, 255),          # #1f2328 GitHub light text
        cursor_color=(5, 150, 105, 255),
    )

    # Dark theme: white/light text, bright emerald highlight
    create_typing_gif(
        ASSETS_DIR / "title_dark.gif",
        highlight_color=(52, 211, 153, 255),   # #34d399 emerald-400
        text_color=(240, 246, 252, 255),       # #f0f6fc GitHub dark text
        cursor_color=(52, 211, 153, 255),
    )


if __name__ == "__main__":
    main()
