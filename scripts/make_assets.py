"""Draw the repository icon (docs/assets/icon.svg). Standard library only.

Usage:
    python scripts/make_assets.py

The icon is a rounded square in the area color with a simple geometric mark,
a loop from the model to a tool and back; no text and no third-party logo. The
output is byte-stable: running the script twice gives the same file.
"""

import os

# area colors (fill, stroke), shared with docs/diagrams/architecture.d2
FILL, INK = "#EDE4FA", "#6D3FB3"


def mark(ink):
    """SVG elements of the mark, on a 64 x 64 canvas, drawn in `ink`."""
    s = f'stroke="{ink}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" fill="none"'
    f = f'fill="{ink}"'
    return (
        f'<path d="M46 24 A15 15 0 1 0 47 38" {s}/><path d="M41 18 L47 24 L40 28" {s}/>'
        f'<circle cx="32" cy="32" r="5" {f}/>'
    )


def icon():
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 64 64">'
        f'<rect x="2" y="2" width="60" height="60" rx="14" fill="{FILL}" stroke="{INK}" stroke-width="3"/>'
        f"{mark(INK)}</svg>\n"
    )


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "docs", "assets", "icon.svg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(icon())
    print(f"wrote {os.path.relpath(out, root)}")


if __name__ == "__main__":
    main()
