"""可视化：马尔科夫链状态图绘制和动画生成。"""

import subprocess
import json
import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageSequence
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)


def draw_markov_graph(
    model: dict[str, dict[str, float]],
    figsize: tuple[float, float] = (16, 9),
    prob_threshold: float = 0.01,
    label_fmt: str = "{:.2f}",
    layout: str = "spring",
    node_size: int = 1800,
    font_size: int = 20,
    save_path: str | None = None,
) -> tuple[nx.DiGraph, dict]:
    """绘制马尔科夫链转移图。

    Returns:
        (G, pos) 图对象和节点位置。
    """
    prob_model = {}
    for a, row in model.items():
        total = sum(row.values())
        prob_model[a] = {b: v / total for b, v in row.items() if v > 0} if total > 0 else {}

    G = nx.DiGraph()
    nodes = set(prob_model.keys())
    for a in prob_model:
        nodes.update(prob_model[a].keys())
    G.add_nodes_from(nodes)

    edge_labels = {}
    for a, row in prob_model.items():
        for b, p in row.items():
            if p >= prob_threshold:
                G.add_edge(a, b, weight=p)
                edge_labels[(a, b)] = label_fmt.format(p)

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=0.8 / np.sqrt(max(len(G.nodes()), 1)))
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_facecolor("#f8f9fa")

    node_color = "#6c9df2"
    node_edge_color = "#3b6bb1"

    nx.draw_networkx_nodes(
        G, pos, node_size=node_size, node_color=node_color,
        edgecolors=node_edge_color, linewidths=2,
    )
    nx.draw_networkx_labels(
        G, pos, font_size=font_size, font_color="#ffffff", font_weight="bold",
    )

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    if weights:
        max_w = max(weights)
        scale_factor = 4 / max_w if max_w > 0 else 1
        widths = [w * scale_factor for w in weights]
        alphas = [0.4 + 0.6 * (w / max_w) for w in weights]
    else:
        widths = 1
        alphas = 0.7

    edge_colors = [(0.2, 0.2, 0.2, a) for a in alphas]

    nx.draw_networkx_edges(
        G, pos, arrowstyle="-|>", arrowsize=12, width=widths,
        edge_color=edge_colors, connectionstyle="arc3,rad=0.18",
        min_source_margin=node_size * 0.01, min_target_margin=node_size * 0.01,
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=font_size / 2,
        font_color="#333333", rotate=False,
    )

    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Graph saved: %s", save_path)
    else:
        plt.show()

    return G, pos


def _load_gif_frames(path: str, zoom: float = 0.15) -> list[OffsetImage]:
    """加载 GIF 的每一帧为 OffsetImage。"""
    gif = Image.open(path)
    frames = []
    for frame in ImageSequence.Iterator(gif):
        frames.append(OffsetImage(np.asarray(frame.convert("RGBA")), zoom=zoom))
    return frames


def _draw_static_background(
    ax: plt.Axes,
    G: nx.DiGraph,
    pos: dict,
    prob_threshold: float = 0.01,
    label_fmt: str = "{:.2f}",
    node_size: int = 1800,
    font_size: int = 20,
) -> list:
    """绘制静态背景图（节点 + 边 + 边标签）。"""
    ax.set_facecolor("#f8f9fa")
    ax.axis("off")

    nx.draw_networkx_nodes(
        G, pos, node_size=node_size, node_color="#6c9df2",
        edgecolors="#3b6bb1", linewidths=2, ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos, font_size=font_size, font_color="#ffffff",
        font_weight="bold", ax=ax,
    )

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [w * 4 / max_w for w in weights]
    alphas = [0.4 + 0.6 * (w / max_w) for w in weights]
    edge_colors = [(0.2, 0.2, 0.2, a) for a in alphas]

    edge_artists = nx.draw_networkx_edges(
        G, pos, arrowstyle="-|>", arrowsize=12, width=widths,
        edge_color=edge_colors, connectionstyle="arc3,rad=0.18",
        min_source_margin=node_size * 0.01, min_target_margin=node_size * 0.01, ax=ax,
    )
    if not isinstance(edge_artists, list):
        edge_artists = [edge_artists]

    edge_labels = {}
    label_pos_dict = {}
    for u, v in G.edges():
        w = G[u][v].get("weight", 0)
        if w >= prob_threshold:
            edge_labels[(u, v)] = label_fmt.format(w)
        if G.has_edge(v, u):
            label_pos_dict[(u, v)] = 0.6
            label_pos_dict[(v, u)] = 0.4
        else:
            label_pos_dict[(u, v)] = 0.5

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=font_size / 2,
        font_color="#333333", rotate=False, label_pos=label_pos_dict[(u, v)], ax=ax,
    )
    return edge_artists


def slice_events(
    events: list[dict],
    t0: float,
    t1: float,
) -> list[dict]:
    """时间窗口切片，为每个事件附加 prev_state。

    Args:
        events: 事件列表。
        t0: 窗口开始。
        t1: 窗口结束。

    Returns:
        切片后的事件（时间已相对于 t0）。
    """
    sliced = []
    last_state_per_track: dict[int, str] = {}

    for e in events:
        if e["end"] <= t0 or e["start"] >= t1:
            continue

        track = e["track"]
        state = e["state"]
        prev_state = last_state_per_track.get(track)

        sliced.append({
            **e,
            "start": max(e["start"], t0) - t0,
            "end": min(e["end"], t1) - t0,
            "prev_state": prev_state,
        })
        last_state_per_track[track] = state

    return sliced


def animate_markov_track(
    G: nx.DiGraph,
    pos: dict,
    events: list[dict],
    track_id: int,
    out_path: str,
    gif_path: str | None = None,
    fps: int = 30,
    figsize: tuple[float, float] = (16, 9),
    gif_offset: float = 0.1,
) -> None:
    """为单个轨道生成马尔科夫链动画视频。

    Args:
        G: 马尔科夫图。
        pos: 节点位置。
        events: 事件列表（含 prev_state）。
        track_id: 轨道 ID。
        out_path: 输出 .mp4 路径。
        gif_path: 可选 GIF 叠加。
        fps: 帧率。
        figsize: 画布尺寸。
        gif_offset: GIF 的 Y 轴偏移。
    """
    track_events = [e for e in events if e["track"] == track_id]
    if not track_events:
        logger.warning("Track %d: no events", track_id)
        return

    duration = max(e["end"] for e in track_events)
    n_frames = int(np.ceil(duration * fps))
    times = np.arange(n_frames) / fps

    gif_frames = _load_gif_frames(gif_path) if gif_path else []

    fig, ax = plt.subplots(figsize=figsize)
    edge_artists = _draw_static_background(ax, G, pos)

    active_nodes = nx.draw_networkx_nodes(
        G, pos, node_size=1800, node_color="#E0E0E0", ax=ax,
    )
    edge_index = {(u, v): i for i, (u, v) in enumerate(G.edges())}

    active_edges: set = set()
    gif_boxes: list = []

    pbar = tqdm(total=n_frames, desc="Rendering frames")

    def update(frame_idx: int):
        nonlocal active_edges, gif_boxes
        t = times[frame_idx]

        current_events = [e for e in track_events if e["start"] <= t < e["end"]]
        states = {e["state"] for e in current_events}
        active_nodes.set_color([
            "#D62728" if n in states else "#6c9df2" for n in G.nodes()
        ])

        for u, v in active_edges:
            edge_artists[edge_index[(u, v)]].set_color((0.2, 0.2, 0.2, 0.3))
        active_edges.clear()

        for e in current_events:
            u = e.get("prev_state")
            v = e["state"]
            if u is not None and (u, v) in edge_index:
                edge_artists[edge_index[(u, v)]].set_color("#D62728")
                active_edges.add((u, v))

        for box in gif_boxes:
            box.remove()
        gif_boxes.clear()

        if gif_frames:
            for e in current_events:
                s = e["state"]
                if s in pos:
                    frame = gif_frames[frame_idx % len(gif_frames)]
                    x, y = pos[s]
                    ab = AnnotationBbox(frame, (x, y + gif_offset), frameon=False, zorder=10)
                    ax.add_artist(ab)
                    gif_boxes.append(ab)

        pbar.update(1)
        return []

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    ani.save(out_path, fps=fps)
    plt.close(fig)
    pbar.close()
    logger.info("Animation saved: %s", out_path)


def mux_video_audio(video_path: str, audio_path: str, out_path: str) -> None:
    """用 ffmpeg 合成视频+音频。"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    logger.info("Muxed: %s", out_path)
