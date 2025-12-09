import cv2


def draw_hud(frame, text_lines):
    """
    Draws text overlay on a video frame.
    text_lines: List of strings to draw.
    """
    if not frame.flags.writeable:
        frame = frame.copy()

    # Constants
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4  # Smaller font
    thickness = 1
    color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black

    # Starting position
    x, y = 8, 16
    line_height = 16  # Tighter spacing

    for line in text_lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)

        # Background box
        cv2.rectangle(frame, (x - 2, y - h - 2), (x + w + 2, y + 2), bg_color, -1)

        # Text
        cv2.putText(frame, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        y += line_height

    return frame