#!/usr/bin/env python3
"""Stitch unified Forge Protocol 4-sheet review bundles from a single capture folder."""
from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path("/Users/ashutosh/development-work/g4-neon/motionbricks-web")


def fit_tile(img_path: Path, w: int, h: int, label: str) -> Image.Image:
    header_h = 28
    canvas = Image.new("RGB", (w, h + header_h), (12, 15, 20))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w, header_h], fill=(18, 22, 30))
    draw.text((12, 7), label, fill=(235, 242, 250))
    if img_path.exists():
        im = Image.open(img_path).convert("RGB")
        im.thumbnail((w, h), Image.Resampling.LANCZOS)
        ox = (w - im.width) // 2
        oy = header_h + (h - im.height) // 2
        canvas.paste(im, (ox, oy))
    return canvas


def make_grid(tiles: list[tuple[Path, str]], cols: int, tile_w: int, tile_h: int, out_path: Path) -> None:
    rows = (len(tiles) + cols - 1) // cols
    cell_h = tile_h + 28
    sheet = Image.new("RGB", (cols * tile_w, rows * cell_h), (12, 15, 20))
    for i, (p, lbl) in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(fit_tile(p, tile_w, tile_h, lbl), (c * tile_w, r * cell_h))
    sheet.save(out_path)
    print(f"Wrote {out_path} ({sheet.width}x{sheet.height})")


def main() -> None:
    cap_dir = Path(sys.argv[1]).resolve()
    target = ROOT / "targets/t01.png"

    # Sheet 1: Wide & Target Side-by-Side + Daylight Gallery + 1280x720 Laptop
    make_grid(
        [
            (target, "TARGET: targets/t01.png (North Star)"),
            (cap_dir / "01_hero_dark.png", "LIVE HERO: cam-01 Dark Pavilion (1440x900)"),
            (cap_dir / "02_hero_light.png", "LIVE THEME: Warm Daylight Gallery (1440x900)"),
            (cap_dir / "03_laptop_1280.png", "LIVE LAPTOP: 1280x720 + PS5 Exclusive HUD"),
        ],
        cols=2,
        tile_w=960,
        tile_h=600,
        out_path=cap_dir / "sheet_01_wide_target.png",
    )

    # Sheet 2: PS5 Exclusive States & Interactive Button Causality
    make_grid(
        [
            (cap_dir / "04_ps5_square_jab.png", "PS5 EXCLUSIVE: Square -> Left Jab + Haptics (Dark)"),
            (cap_dir / "05_ps5_r2_cross_light.png", "PS5 EXCLUSIVE: R2 -> Right Cross + L2 Slow-Mo (Light)"),
            (cap_dir / "10_pose_xray.png", "PS5 TOUCHPAD: Neural X-Ray Line-Art Skeleton"),
            (cap_dir / "06_dressed_closeup.png", "CLOSE-UP: Dressed SOMA Athlete (cam-02)"),
        ],
        cols=2,
        tile_w=960,
        tile_h=600,
        out_path=cap_dir / "sheet_02_ps5_closeup.png",
    )

    # Sheet 3: Extreme Action Poses Deformation & Parkour QC
    make_grid(
        [
            (cap_dir / "07_pose_vault.png", "ACTION: Parkour Vault Clearing Concrete/Copper Plinth"),
            (cap_dir / "08_pose_duck.png", "ACTION: Overhead Duck (PS5 D-Pad Down -> Overhead Duck)"),
            (cap_dir / "09_pose_sit.png", "ACTION: Seated on Minimalist Gallery Bench"),
            (cap_dir / "13_stress_gauntlet.png", "ARENA: Rear Stress-Test Gauntlet (Hurdle -> Vault -> Duck)"),
        ],
        cols=2,
        tile_w=960,
        tile_h=600,
        out_path=cap_dir / "sheet_03_actions_qc.png",
    )

    # Sheet 4: More Popover Specialty Styles/Emotes + System Overview Help Modal
    make_grid(
        [
            (cap_dir / "12_more_popover_emote.png", "DOCK: More Popover (Specialty Styles + Active Stretch Emote)"),
            (cap_dir / "11_help_modal.png", "HELP: System Overview & Universal 3D OBB Architecture Modal"),
        ],
        cols=2,
        tile_w=960,
        tile_h=600,
        out_path=cap_dir / "sheet_04_arena_help.png",
    )


if __name__ == "__main__":
    main()
