import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageSequence
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import json

lyrics_path = "./output/tchop35a/tchop35a_lyrics.json"
out_path="./output/tchop35a/"
gif_path="./assets/animation/帝宝挥拳加油.gif"
model_path="./models/character_markov/"

def draw(model, figsize=(16,9), prob_threshold=0.01,
                label_fmt="{:.2f}", layout="spring",
                node_size=1800, font_size=11, save_path=None):

    # --------- Normalize Transition Matrix ---------
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    prob_model = {}
    for a, row in model.items():
        total = sum(row.values())
        if total == 0:
            prob_model[a] = {}
        else:
            prob_model[a] = {b: v / total for b, v in row.items() if v > 0}

    # --------- Build Graph ---------
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

    # --------- Layout ↑ Slightly More Beautiful Version ---------
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=0.8/np.sqrt(len(G.nodes())))
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
    ax.set_facecolor("#f8f9fa")  # 柔和背景色

    # --------- Node Colors (soft blue scheme) ---------
    node_color = "#6c9df2"       # 主节点颜色
    node_edge_color = "#3b6bb1"  # 节点边框

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_color,
        edgecolors=node_edge_color,
        linewidths=2
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=font_size,
        font_color="#ffffff",  # 白字更干净
        font_weight="bold"
    )

    # --------- Edge Styling ---------
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    if weights:
        max_w = max(weights)
        scale_factor = 4 / max_w if max_w > 0 else 1
        widths = [w * scale_factor for w in weights]
        alphas = [0.4 + 0.6*(w/max_w) for w in weights]  # 越大越不透明
    else:
        widths = 1
        alphas = 0.7

    edge_colors = [(0.2, 0.2, 0.2, a) for a in alphas]  # 深灰 → 黑，透明度随权重变化

    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='-|>',
        arrowsize=12,
        width=widths,
        edge_color=edge_colors,
        connectionstyle='arc3,rad=0.18',
        min_source_margin=node_size * 0.01,
        min_target_margin=node_size * 0.01
    )

    # --------- Edge Labels ---------
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=font_size-1,
        font_color="#333333",
        rotate=False
    )

    plt.axis("off")
    plt.tight_layout()

    # --------- Save or Show ---------
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return G, pos

def load_gif_frames(path, zoom=0.15):
    gif = Image.open(path)
    frames = []
    for frame in ImageSequence.Iterator(gif):
        frames.append(
            OffsetImage(np.asarray(frame.convert("RGBA")), zoom=zoom)
        )
    return frames

def slice_events(events, t0, t1):
    """
    events: List[Event]
    t0, t1: 时间窗口（秒）

    return:
        sliced_events: List[Event]
    """
    sliced = []

    for e in events:
        if e["end"] <= t0 or e["start"] >= t1:
            continue

        sliced.append({
            **e,
            "start": max(e["start"], t0) - t0,
            "end": min(e["end"], t1) - t0
        })

    return sliced

def animate_markov_track(
    G,
    pos,
    events,
    track_id,
    out_path,
    gif_path=None,
    fps=30
):
    # -----------------------
    # 过滤该轨道 events
    # -----------------------
    track_events = [e for e in events if e["track"] == track_id]
    if not track_events:
        print(f"Track {track_id}: no events, skipped")
        return

    t_start = min(e["start"] for e in track_events)
    t_end   = max(e["end"] for e in track_events)
    times = np.linspace(t_start, t_end, int((t_end - t_start) * fps))

    gif_frames = load_gif_frames(gif_path)
    gif_boxes = []

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    # -----------------------
    # 绘制静态背景
    # -----------------------
    nx.draw_networkx_nodes(G, pos, node_color="#E0E0E0", node_size=1400, ax=ax)
    edge_collection = nx.draw_networkx_edges(G, pos, edge_color="#B0B0B0", arrows=True, ax=ax)
    if not isinstance(edge_collection, list):
        edge_collection = [edge_collection]  # 确保是 list
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    # -----------------------
    # 动态节点
    # -----------------------
    active_nodes = nx.draw_networkx_nodes(G, pos, node_color="#D62728", node_size=1600, ax=ax)

    prev_states = set()

    def update(frame_idx):
        nonlocal prev_states, gif_boxes
        t = times[frame_idx]

        # 清理旧 GIF
        for box in gif_boxes:
            box.remove()
        gif_boxes = []

        # 当前激活状态
        current_events = [e for e in track_events if e["start"] <= t < e["end"]]
        states = {e["state"] for e in current_events}

        # -----------------------
        # 节点高亮
        # -----------------------
        node_colors = ["#D62728" if n in states else "#E0E0E0" for n in G.nodes()]
        active_nodes.set_color(node_colors)

        # -----------------------
        # 边高亮（逐条修改颜色）
        # -----------------------
        for idx, (u, v) in enumerate(G.edges()):
            color = "#D62728" if u in prev_states and v in states else "#B0B0B0"
            if idx < len(edge_collection):
                edge_collection[idx].set_color(color)

        # -----------------------
        # GIF 叠加
        # -----------------------
        if gif_frames:
            frame = gif_frames[frame_idx % len(gif_frames)]
            for s in states:
                if s in pos:
                    ab = AnnotationBbox(frame, pos[s], frameon=False, zorder=10)
                    ax.add_artist(ab)
                    gif_boxes.append(ab)

        prev_states = states
        return []

    ani = FuncAnimation(fig, update, frames=len(times), interval=1000/fps)
    ani.save(out_path, fps=fps)
    plt.close(fig)
    print(f"Saved track {track_id} animation → {out_path}")



with open(lyrics_path, "r", encoding="utf-8") as f:
    events = json.load(f)

with open(model_path + "哈基米only.json", "r", encoding="utf-8") as f:
    model = json.load(f)

G, pos = draw(model, save_path=out_path + "track0.png")

slice_events = slice_events(events, t0=0.0, t1=30.0)
animate_markov_track(
    G=G,
    pos=pos,
    events=slice_events,
    track_id=1,
    out_path="./output/tchop35a/track0.mp4",
    gif_path="./assets/animation/帝宝挥拳加油.gif",
    fps=30
)