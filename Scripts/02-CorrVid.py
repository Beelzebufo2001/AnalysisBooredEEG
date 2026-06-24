#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

import config
from helpers_corr import (
    ensure_dir_exists,
    get_valid_subjects,
    load_connectivity,
    extract_times_from_files,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Create video from correlation matrices.")
    parser.add_argument("--subject", nargs="+", default=None,
                        help="One or more subjects, e.g. C01 C02. Omit for all.")
    parser.add_argument("--input-dir", type=Path, default=config.CORR_OUTPUT_DIR,
                        help="Directory with corr matrix folders")
    parser.add_argument("--output-dir", type=Path, default=config.VIDEO_OUTPUT_DIR,
                        help="Directory for output videos")
    parser.add_argument("--fps", type=int, default=5,
                        help="Frames per second")
    parser.add_argument("--vmin", type=float, default=-1.0,
                        help="Min colormap value")
    parser.add_argument("--vmax", type=float, default=1.0,
                        help="Max colormap value")
    parser.add_argument("--cmap", type=str, default="coolwarm",
                        help="Matplotlib colormap")
    return parser.parse_args()


def make_video(subject, input_dir, output_dir, fps, vmin, vmax, cmap):
    connectivity, files = load_connectivity(subject, input_dir)
    times = extract_times_from_files(files)

    ensure_dir_exists(output_dir)
    out_path = Path(output_dir) / f"{subject}_corr_video.mp4"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(connectivity[0], vmin=vmin, vmax=vmax, cmap=cmap, animated=True)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    title = ax.set_title(f"{subject} | t = {times[0]} s")

    def update(frame):
        im.set_array(connectivity[frame])
        title.set_text(f"{subject} | t = {times[frame]} s")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(connectivity), interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=150)
    plt.close(fig)

    print(f"Saved video: {out_path}")


def main():
    args = parse_args()
    subjects = args.subject if args.subject is not None else "all"
    subjects = get_valid_subjects(subjects, args.input_dir)

    for subject in subjects:
        make_video(
            subject=subject,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            fps=args.fps,
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
        )


if __name__ == "__main__":
    main()