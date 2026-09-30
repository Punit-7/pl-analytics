def channel(c: int) -> float:
    """Convert one 0-255 colour channel to linear light."""
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    light, dark = sorted([luminance(a), luminance(b)], reverse=True)
    return (light + 0.05) / (dark + 0.05)


SURFACE = "#1E0A2A"
for text in ["#FFFFFF", "#C9BFD4", "#04F5FF", "#00FF85", "#E90052"]:
    ratio = contrast(text, SURFACE)
    verdict = "OK for all text" if ratio >= 4.5 else "large text only" if ratio >= 3 else "fail"
    print(f"{text} on {SURFACE}: {ratio:.1f}:1  {verdict}")
