#!/usr/bin/env python3
"""Compose agentview videos with overlaid reward curves and success annotations.

Reads precomputed predictions under ``<cache_root>/predictions/`` (see ``run_reward_inference.py``)
and RGB frames from ``<cache>/.../frames.npz`` — no model inference.

The composite figure is fixed to **512 px** wide (matches ``run_trained_agent``-style exports); height
scales linearly with the cached aspect ratio. Frames are resized with Lanczos when native width differs.

Default selection matches the eval cache layout: one ``expert_ph`` demo, two ``rollout_success``
rollouts, and two ``rollout_failure`` rollouts (five MP4s). Override with ``--demos``.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("plot_reward_trajectory_videos")

# Success strip (was 7.5pt; user requested 1.5×); display uses 0.8× of this cap below
SUCCESS_LABEL_FONTSIZE = 7.5 * 1.5
SUCCESS_STRIP_FONT_SCALE = 0.8

# Fixed output width (pixels); height follows demo aspect when resizing frames
OUTPUT_WIDTH_PX = 512

# Progress curves (solid)
COLOR_TOPREWARD = "#eb0707"
COLOR_ROBODOPAMINE = "#0729eb"
COLOR_RBM = "#38eb07"

# Ground truth overlays (dotted)
COLOR_GT_LINEAR = "#0f0f0f"
COLOR_GT_SIM = "#f58402"

# Success-prediction text (match progress family where applicable)
TEXT_COLOR_SUCCESS_DET = "#f1f502"
TEXT_COLOR_TOPREWARD = COLOR_TOPREWARD
TEXT_COLOR_ROBOREWARD = "#9802f5"
TEXT_COLOR_RBM = COLOR_RBM

PRED_SUBDIRS = {
    "topreward": "topreward_qwen",
    "robodopamine": "robodopamine",
    "robometer": "rbm",
    "roboreward": "roboreward",
    "success_detector": "success_detector",
}


def load_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_row_for(
    rows: list[dict[str, str]], dataset_name: str, demo_key: str
) -> dict[str, str] | None:
    for r in rows:
        if r.get("dataset_name") == dataset_name and r.get("demo_key") == demo_key:
            return r
    return None


def normalize_to_unit_interval(x: np.ndarray) -> np.ndarray:
    """Map a 1D series into [0, 1]: clip if already in-range, else min–max."""
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        return a
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.zeros_like(a)
    if lo >= -1e-6 and hi <= 1.0 + 1e-6 and hi - lo <= 1.0 + 1e-6:
        return np.clip(a, 0.0, 1.0)
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _min_len(*arrays: np.ndarray) -> int:
    return min(int(a.shape[0]) for a in arrays if a is not None and a.size)


def trim_to_T(*arrays: np.ndarray, t: int) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(a, dtype=np.float64).ravel()[:t] for a in arrays)


def load_rgb(cache_root: Path, dataset_name: str, demo_key: str) -> np.ndarray:
    p = cache_root / dataset_name / demo_key / "frames.npz"
    z = np.load(p)
    rgb = np.asarray(z["rgb"], dtype=np.uint8)
    if rgb.ndim != 4 or rgb.shape[-1] < 3:
        raise ValueError(f"Bad rgb in {p}: {rgb.shape}")
    return rgb[..., :3]


def resize_frames_to_width_px(rgb: np.ndarray, width_px: int) -> np.ndarray:
    """Resize each frame to ``width_px`` wide; height follows native aspect ratio."""
    if rgb.size == 0:
        return rgb
    h0, w0 = int(rgb.shape[1]), int(rgb.shape[2])
    tw = int(width_px)
    th = max(1, int(round(h0 * (tw / float(w0)))))
    if tw == w0 and th == h0:
        return rgb

    try:
        import cv2  # type: ignore

        interp = getattr(cv2, "INTER_LANCZOS4", cv2.INTER_CUBIC)
        out = np.empty((rgb.shape[0], th, tw, 3), dtype=np.uint8)
        for i in range(rgb.shape[0]):
            out[i] = cv2.resize(rgb[i], (tw, th), interpolation=interp)
        return out
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore

        out = np.empty((rgb.shape[0], th, tw, 3), dtype=np.uint8)
        for i in range(rgb.shape[0]):
            im = Image.fromarray(rgb[i])
            im = im.resize((tw, th), Image.Resampling.LANCZOS)
            out[i] = np.asarray(im)
        return out
    except Exception as e:
        raise SystemExit(
            "resize_frames_to_width_px needs opencv-python or pillow. " f"{e!r}"
        ) from e


def load_pred_npz(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    z = np.load(path)
    return {k: np.asarray(z[k]) for k in z.files}


def traj_success_01(z: dict[str, Any]) -> float | None:
    if "traj_success_pred" not in z:
        return None
    v = float(np.asarray(z["traj_success_pred"]).ravel()[0])
    if v != v:  # NaN
        return None
    return 1.0 if v >= 0.5 else 0.0


def roboreward_discrete_bin_from_pred(pred: np.ndarray) -> int | None:
    """Invert RoboReward normalized broadcast to discrete score 1–5 (see ``backends``)."""
    a = np.asarray(pred, dtype=np.float64).ravel()
    if a.size == 0:
        return None
    pv = float(a[0])
    if pv != pv:
        return None
    return int(np.clip(np.round((pv + 0.25) * 4.0), 1, 5))


def timing_frame(z: dict[str, Any], t_full: int) -> float | None:
    if "success_pred_timing_frame" not in z:
        return None
    v = float(np.asarray(z["success_pred_timing_frame"]).ravel()[0])
    if v != v:
        return None
    return v


def _unclip_y_axis_text(ax: Any) -> None:
    """Tick/ylabel artists often use clip_on=True and get cropped at the axes bbox."""
    import matplotlib.pyplot as plt

    ylab = ax.yaxis.label
    if ylab is not None:
        ylab.set_clip_on(False)
    plt.setp(ax.get_yticklabels(), clip_on=False)
    for t in ax.yaxis.get_major_ticks():
        if getattr(t, "label1", None) is not None:
            t.label1.set_clip_on(False)
        if getattr(t, "label2", None) is not None:
            t.label2.set_clip_on(False)
    off = ax.yaxis.get_offset_text()
    if off is not None:
        off.set_clip_on(False)


def annotation_start_frame(
    z_tr: dict[str, Any],
    z_rbm: dict[str, Any],
    z_sd: dict[str, Any],
    t_full: int,
) -> int:
    """First frame at which to show success annotations: max timing among heads that expose it.

    TopReward / Robometer / success_detector save ``success_pred_timing_frame`` aligned to manifest
    length. RoboReward is end-of-episode only (no per-frame timing); it does not participate here,
    so we do not force the strip to the last frame unless no other timing exists.
    """
    cands: list[float] = []
    for z in (z_tr, z_rbm, z_sd):
        tf = timing_frame(z, t_full)
        if tf is not None:
            cands.append(tf)
    if not cands:
        return max(0, t_full - 1)
    return int(np.clip(int(np.ceil(max(cands))), 0, t_full - 1))


def render_video_for_demo(
    *,
    cache_root: Path,
    rows: list[dict[str, str]],
    dataset_name: str,
    demo_key: str,
    out_path: Path,
    fps: int,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = manifest_row_for(rows, dataset_name, demo_key)
    if row is None:
        raise SystemExit(f"No manifest row for {dataset_name}/{demo_key}")

    rgb = load_rgb(cache_root, dataset_name, demo_key)
    t_rgb = int(rgb.shape[0])
    rgb_display = resize_frames_to_width_px(rgb, OUTPUT_WIDTH_PX)

    paths = {
        "topreward": cache_root / "predictions" / PRED_SUBDIRS["topreward"] / dataset_name / f"{demo_key}.npz",
        "robodopamine": cache_root / "predictions" / PRED_SUBDIRS["robodopamine"] / dataset_name / f"{demo_key}.npz",
        "rbm": cache_root / "predictions" / PRED_SUBDIRS["robometer"] / dataset_name / f"{demo_key}.npz",
        "roboreward": cache_root / "predictions" / PRED_SUBDIRS["roboreward"] / dataset_name / f"{demo_key}.npz",
        "success_detector": cache_root / "predictions" / PRED_SUBDIRS["success_detector"] / dataset_name / f"{demo_key}.npz",
    }
    loaded: dict[str, dict[str, Any]] = {}
    for k, p in paths.items():
        loaded[k] = load_pred_npz(p)

    z0 = loaded["topreward"]
    t = _min_len(
        rgb[:, 0, 0, 0],
        z0["pred"],
        loaded["robodopamine"]["pred"],
        loaded["rbm"]["pred"],
        loaded["roboreward"]["pred"],
        z0["gt_linear_time"],
        z0["gt_cumulative_normalized"],
    )
    if t != t_rgb:
        logger.warning("Trimming T %d -> %d for %s/%s", t_rgb, t, dataset_name, demo_key)
        rgb = rgb[:t]
        rgb_display = rgb_display[:t]

    pred_tr, pred_rd, pred_rbm, pred_rr = trim_to_T(
        loaded["topreward"]["pred"],
        loaded["robodopamine"]["pred"],
        loaded["rbm"]["pred"],
        loaded["roboreward"]["pred"],
        t=t,
    )
    gt_lin, gt_cum = trim_to_T(z0["gt_linear_time"], z0["gt_cumulative_normalized"], t=t)

    y_tr = normalize_to_unit_interval(pred_tr)
    y_rd = normalize_to_unit_interval(pred_rd)
    y_rbm = normalize_to_unit_interval(pred_rbm)
    y_glin = normalize_to_unit_interval(gt_lin)
    y_gcum = normalize_to_unit_interval(gt_cum)

    x = np.arange(t, dtype=np.float64)
    t_show = annotation_start_frame(
        loaded["topreward"],
        loaded["rbm"],
        loaded["success_detector"],
        t,
    )

    succ_tr = traj_success_01(loaded["topreward"])
    succ_rr = traj_success_01(loaded["roboreward"])
    succ_rbm = traj_success_01(loaded["rbm"])
    succ_sd = traj_success_01(loaded["success_detector"])
    rr_bin = roboreward_discrete_bin_from_pred(pred_rr)

    gt_succ = int(row.get("success_label") or row.get("success_label".upper()) or 0)
    try:
        gt_succ = int(gt_succ)
    except (TypeError, ValueError):
        gt_succ = 0

    title = f"{dataset_name} {demo_key}  |  GT success={gt_succ}  |  split={row.get('split_tag', '')}"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Figure width in inches so rasterized canvas is exactly OUTPUT_WIDTH_PX wide.
    # Manual axes positions (figure coordinates): video + success strip use full content width
    # (same left/right as each other). The reward plot uses a larger *left* inset so y-axis text
    # fits, but the same *right* edge as the video — avoids a 2-column GridSpec where the plot
    # only occupied ~76% width with empty space on the right.
    fig_w_in = OUTPUT_WIDTH_PX / float(dpi)
    fig_h_in = 6.0
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    # Gaps in figure-height fraction: px / (inches * dpi)
    px_to_fig_h = 1.0 / (fig_h_in * float(dpi))
    gap_video_plot_px = 5.0
    # Space between plot x-axis (ticks + "frame") and success strip; strip text is anchored low.
    gap_plot_strip_px = 22.0
    gap_vp = gap_video_plot_px * px_to_fig_h
    gap_ps = gap_plot_strip_px * px_to_fig_h

    r_vid, r_plot, r_strip = 2.65, 2.45, 0.48
    r_sum = r_vid + r_plot + r_strip
    bottom_m, top_m = 0.05, 0.96
    span = top_m - bottom_m - gap_vp - gap_ps
    h_strip = span * (r_strip / r_sum)
    h_plot = span * (r_plot / r_sum)
    h_vid = span * (r_vid / r_sum)
    # Bottom → top: success strip, gap, plot, gap, video
    y_strip = bottom_m
    y_plot = y_strip + h_strip + gap_ps
    y_vid = y_plot + h_plot + gap_vp
    # Full-width panels (match left/right so image and strip align with plot's right edge)
    lw, rw = 0.02, 0.98
    fw = rw - lw
    ax_strip = fig.add_axes([lw, y_strip, fw, h_strip])
    # Plot: left inset for y ticks + label (two-line label is narrower than one long line)
    plot_left, plot_w = 0.095, rw - 0.095
    ax_plot = fig.add_axes([plot_left, y_plot, plot_w, h_plot])
    ax_vid = fig.add_axes([lw, y_vid, fw, h_vid])

    ax_strip.set_axis_off()
    ax_strip.set_navigate(False)
    ax_strip.set_xlim(0.0, 1.0)
    ax_strip.set_ylim(0.0, 1.0)

    frames: list[np.ndarray] = []

    def yes_no(v: float | None) -> str:
        if v is None:
            return "n/a"
        return "success" if v >= 0.5 else "fail"

    rr_bin_str = f"{rr_bin} (1-5)" if rr_bin is not None else "n/a (1-5)"
    # Fixed horizontal slots (transAxes x-centers), left → right
    ann_slots: list[tuple[str, str]] = [
        (f"GT (manifest)\n{yes_no(float(gt_succ))}", "#000000"),
        (f"Success detector\n{yes_no(succ_sd)}", TEXT_COLOR_SUCCESS_DET),
        (f"TopReward\n{yes_no(succ_tr)}", TEXT_COLOR_TOPREWARD),
        (f"RoboReward\n{yes_no(succ_rr)}\n{rr_bin_str}", TEXT_COLOR_ROBOREWARD),
        (f"Robometer\n{yes_no(succ_rbm)}", TEXT_COLOR_RBM),
    ]
    slot_x = [0.1, 0.3, 0.5, 0.7, 0.9]

    for i in range(t):
        ax_vid.clear()
        ax_plot.clear()
        ax_strip.clear()
        ax_strip.set_axis_off()
        ax_strip.set_xlim(0.0, 1.0)
        ax_strip.set_ylim(0.0, 1.0)

        # aspect='auto' fills the video axes; 'equal' letterboxes in tall cells (looks off-center).
        ax_vid.imshow(rgb_display[i], aspect="auto", interpolation="bilinear")
        ax_vid.axis("off")
        ax_vid.set_title(title, fontsize=7, pad=0)

        ax_plot.plot(x, y_tr, color=COLOR_TOPREWARD, linewidth=1.3, label="TopReward", zorder=4)
        ax_plot.plot(x, y_rd, color=COLOR_ROBODOPAMINE, linewidth=1.3, label="RoboDopamine", zorder=4)
        ax_plot.plot(x, y_rbm, color=COLOR_RBM, linewidth=1.3, label="Robometer", zorder=4)
        ax_plot.plot(
            x,
            y_glin,
            color=COLOR_GT_LINEAR,
            linewidth=1.3,
            linestyle=":",
            label="GT linear time",
            zorder=2,
        )
        ax_plot.plot(
            x,
            y_gcum,
            color=COLOR_GT_SIM,
            linewidth=1.3,
            linestyle=":",
            label="GT sim reward",
            zorder=2,
        )
        ax_plot.axvline(i, color="#333333", linewidth=1.0, alpha=0.85, zorder=5)
        ax_plot.set_xlim(0, max(t - 1, 0))
        ax_plot.set_ylim(0.0, 1.0)
        ax_plot.set_xlabel("frame", labelpad=4, fontsize=7)
        ax_plot.set_ylabel(
            "normalized reward",
            labelpad=4,
            fontsize=6,
            linespacing=0.85,
        )
        ax_plot.tick_params(axis="x", labelsize=6)
        ax_plot.tick_params(axis="y", labelsize=6, pad=2)
        # Upper left keeps the legend off the late-trajectory curves (lower right overlapped data).
        ax_plot.legend(
            loc="upper left",
            fontsize=5,
            framealpha=0.95,
            fancybox=False,
            borderpad=0.4,
            labelspacing=0.35,
            handlelength=1.6,
        )
        ax_plot.grid(True, alpha=0.25)
        ax_plot.spines["top"].set_visible(False)
        ax_plot.spines["right"].set_visible(False)
        _unclip_y_axis_text(ax_plot)

        strip_fs = min(SUCCESS_LABEL_FONTSIZE, 9.0) * SUCCESS_STRIP_FONT_SCALE
        if i >= t_show:
            # Anchor near the figure bottom so labels stay under the plot's x-axis, not under "frame".
            for x_c, (text, color) in zip(slot_x, ann_slots):
                ax_strip.text(
                    x_c,
                    0.06,
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=strip_fs,
                    color=color,
                    transform=ax_strip.transAxes,
                )

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        if w != OUTPUT_WIDTH_PX:
            logger.warning("Canvas width %d != OUTPUT_WIDTH_PX %d (dpi=%s)", w, OUTPUT_WIDTH_PX, dpi)
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())

    plt.close(fig)

    _write_mp4_rgb_frames(frames, out_path, fps)
    logger.info("Wrote %s (%d frames)", out_path, len(frames))


def _write_mp4_rgb_frames(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    """Write (T,H,W,3) uint8 RGB MP4 via imageio (preferred) or OpenCV fallback."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errs: list[str] = []
    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(out_path), frames, fps=fps, codec="libx264")
        return
    except Exception as e:
        errs.append(f"imageio: {e!r}")
    try:
        import cv2  # type: ignore

        if not frames:
            raise ValueError("no frames")
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
        if not vw.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        for fr in frames:
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            vw.write(bgr)
        vw.release()
        return
    except Exception as e:
        errs.append(f"opencv: {e!r}")
    raise SystemExit(
        "Could not write MP4 (tried imageio then opencv). Errors: " + " | ".join(errs)
    )


def default_demo_specs() -> list[tuple[str, str]]:
    """1 expert + 2 rollout successes + 2 rollout failures (distinct rollouts)."""
    return [
        ("expert_ph", "demo_0"),
        ("rollout_mixed", "demo_0"),
        ("rollout_mixed", "demo_3"),
        ("rollout_mixed", "demo_2"),
        ("rollout_mixed", "demo_5"),
    ]


def parse_demo_args(specs: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in specs:
        if ":" not in s:
            raise SystemExit(f"Bad --demo {s!r}; use dataset:demo_key e.g. expert_ph:demo_0")
        ds, dk = s.split(":", 1)
        out.append((ds.strip(), dk.strip()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-root",
        type=str,
        default="/scratch/bggq/asunesara/reward_eval_cache",
        help="Directory with manifest.csv, per-demo cache, and predictions/",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Override manifest path (default: <cache-root>/manifest.csv)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for MP4 files (default: <repo>/reward_traj_videos)",
    )
    p.add_argument(
        "--demo",
        dest="demos",
        action="append",
        default=[],
        help="dataset:demo_key (repeat). Default: 1 expert + 2 success + 2 failure rollouts.",
    )
    p.add_argument("--fps", type=int, default=20)
    p.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Rasterization DPI; figure width in inches is 512/dpi so output width stays 512 px.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("plot_reward_trajectory_videos").setLevel(
        logging.DEBUG if args.verbose else logging.INFO
    )

    cache_root = Path(args.cache_root).expanduser()
    manifest_path = Path(args.manifest).expanduser() if args.manifest else cache_root / "manifest.csv"
    rows = load_manifest(manifest_path)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else repo_root / "reward_traj_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_specs = parse_demo_args(args.demos) if args.demos else default_demo_specs()

    for dataset_name, demo_key in demo_specs:
        safe = f"{dataset_name}__{demo_key}"
        out_mp4 = out_dir / f"reward_curves_{safe}.mp4"
        logger.info("Rendering %s / %s -> %s", dataset_name, demo_key, out_mp4)
        render_video_for_demo(
            cache_root=cache_root,
            rows=rows,
            dataset_name=dataset_name,
            demo_key=demo_key,
            out_path=out_mp4,
            fps=int(args.fps),
            dpi=int(args.dpi),
        )


if __name__ == "__main__":
    main()
