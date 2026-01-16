import subprocess
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageSequence
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import json
import subprocess
import os
from tqdm import tqdm
import matplotlib.patches as mpatches

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

def states_at_time(events, t):
    return {e["state"] for e in events if e["start"] <= t < e["end"]}

def slice_events(events, t0, t1):
    """
    events: List[Event]，每个 event 至少有 track, state, start, end
    t0, t1: 时间窗口（秒）
    
    return: sliced_events: List[Event]，每个 event 新增 prev_state
    """
    sliced = []
    last_state_per_track = {}  # 记录每条轨道上前一个状态

    for e in events:
        if e["end"] <= t0 or e["start"] >= t1:
            continue

        start = max(e["start"], t0) - t0
        end = min(e["end"], t1) - t0

        track = e["track"]
        state = e["state"]
        prev_state = last_state_per_track.get(track, None)

        sliced.append({
            **e,
            "start": start,
            "end": end,
            "prev_state": prev_state
        })

        last_state_per_track[track] = state

    return sliced

def draw_static_background(ax, G, pos, prob_threshold=0.01, label_fmt="{:.2f}", node_size=1800, font_size=11):
    ax.set_facecolor("#f8f9fa")  # 背景色
    ax.axis("off")

    # --------- Node Colors ---------
    node_color = "#6c9df2"
    node_edge_color = "#3b6bb1"

    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_color,
        edgecolors=node_edge_color,
        linewidths=2,
        ax=ax
    )

    # 绘制节点标签
    nx.draw_networkx_labels(
        G, pos,
        font_size=font_size,
        font_color="#ffffff",
        font_weight="bold",
        ax=ax
    )

    # --------- Edge Styling ---------
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [w * 4 / max_w for w in weights]
    alphas = [0.4 + 0.6*(w/max_w) for w in weights]
    edge_colors = [(0.2, 0.2, 0.2, a) for a in alphas]

    edge_artists = nx.draw_networkx_edges(
        G, pos,
        arrowstyle='-|>',
        arrowsize=12,
        width=widths,
        edge_color=edge_colors,
        connectionstyle='arc3,rad=0.18',
        min_source_margin=node_size * 0.01,
        min_target_margin=node_size * 0.01,
        ax=ax
    )

    if not isinstance(edge_artists, list):
        edge_artists = [edge_artists]

    # --------- Edge Labels ---------
    edge_labels = {}
    label_pos_dict = {}
    for u, v in G.edges():
        w = G[u][v].get("weight", 0)
        if w >= prob_threshold:
            edge_labels[(u, v)] = label_fmt.format(w)

        if G.has_edge(v, u):
            # 双向边，正向偏上，反向偏下
            label_pos_dict[(u, v)] = 0.6  # 正向 label 偏向目标节点
            label_pos_dict[(v, u)] = 0.4  # 反向 label 偏向源节点
        else:
            label_pos_dict[(u, v)] = 0.5  # 单向边，居中

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=font_size-1,
        font_color="#333333",
        rotate=False,
        label_pos=label_pos_dict[(u, v)],
        ax=ax
    )

    return edge_artists

def animate_markov_track(
    G,
    pos,
    events,
    track_id,
    out_path,
    gif_path=None,
    fps=30,
    figsize=(16,9),
    gif_offset=0.1
):
    track_events = [e for e in events if e["track"] == track_id]
    if not track_events:
        print(f"Track {track_id}: no events")
        return

    duration = max(e["end"] for e in track_events)
    n_frames = int(np.ceil(duration * fps))
    times = np.arange(n_frames) / fps

    gif_frames = load_gif_frames(gif_path) if gif_path else []

    fig, ax = plt.subplots(figsize=figsize)
    edge_artists = draw_static_background(ax, G, pos)

    active_nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=1800,
        node_color="#E0E0E0",
        ax=ax
    )

    edge_index = {(u, v): i for i, (u, v) in enumerate(G.edges())}

    active_edges = set()
    gif_boxes = []

    pbar = tqdm(total=n_frames, desc="Rendering frames")

    def update(frame_idx):
        nonlocal active_edges, gif_boxes
        t = times[frame_idx]

        # 当前激活事件
        current_events = [e for e in track_events if e["start"] <= t < e["end"]]

        # 节点高亮
        states = {e["state"] for e in current_events}
        active_nodes.set_color([
            "#D62728" if n in states else "#6c9df2"
            for n in G.nodes()
        ])


        # 先还原上一帧高亮边
        for u, v in active_edges:
            idx = edge_index[(u, v)]
            edge_artists[idx].set_color((0.2,0.2,0.2,0.3))
        active_edges.clear()

        # 高亮当前帧的转移边
        for e in current_events:
            u = e.get("prev_state")
            v = e["state"]
            if u is not None and (u, v) in edge_index:
                idx = edge_index[(u, v)]
                edge_artists[idx].set_color("#D62728")
                active_edges.add((u, v))

        # ---- GIF 叠加 ----
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

    ani = FuncAnimation(
        fig, update, frames=n_frames, interval=1000/fps, blit=False
    )
    ani.save(out_path, fps=fps) 
    plt.close(fig)
    pbar.close()
    print(f"Saved animation → {out_path}")


def mux_video_audio(video_path, audio_path, out_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path
    ]
    subprocess.run(cmd, check=True)

with open(lyrics_path, "r", encoding="utf-8") as f:
    events = json.load(f)

# with open(model_path + "哈基米only.json", "r", encoding="utf-8") as f:
#     model = json.load(f)

# G, pos = draw(model, save_path=out_path + "track1.png")

# slice_events = slice_events(events, t0=0.0, t1=600.0)
# animate_markov_track(
#     G=G,
#     pos=pos,
#     events=slice_events,
#     track_id=1,
#     out_path="./output/tchop35a/track1.mp4",
#     gif_path="./assets/animation/帝宝挥拳加油.gif",
#     fps=30,
#     figsize=(9,16),
#     gif_offset=0.1
# )

# video_path = "./output/tchop35a/track1.mp4"
# audio_path = "./output/tchop35a/tchop35a.wav"
# out_path   = "./output/tchop35a/track1_audio.mp4"

# mux_video_audio(video_path, audio_path, out_path)

# #-------------------------------------------

# with open(model_path + "曼波only.json", "r", encoding="utf-8") as f:
#     model = json.load(f)

# G, pos = draw(model, save_path=out_path + "track2.png")

# slice_events = slice_events(events, t0=0.0, t1=600.0)
# animate_markov_track(
#     G=G,
#     pos=pos,
#     events=slice_events,
#     track_id=2,
#     out_path="./output/tchop35a/track2.mp4",
#     gif_path="./assets/animation/曼波摇.gif",
#     fps=30,
#     figsize=(5, 16),
#     gif_offset=0.13
# )

# video_path = "./output/tchop35a/track2.mp4"
# audio_path = "./output/tchop35a/tchop35a.wav"
# out_path   = "./output/tchop35a/track2_audio.mp4"

# mux_video_audio(video_path, audio_path, out_path)


# #-------------------------------------------

with open(model_path + "哈基米.json", "r", encoding="utf-8") as f:
    model = json.load(f)

G, pos = draw(model, save_path=out_path + "track0.png")

slice_events = slice_events(events, t0=0.0, t1=600)
animate_markov_track(
    G=G,   
    pos=pos,
    events=slice_events,
    track_id=0,
    out_path="./output/tchop35a/track0.mp4",
    gif_path="./assets/animation/帝宝摇.gif",
    fps=30,
    figsize=(10, 16)
)

# video_path = "./output/tchop35a/track0.mp4"
# audio_path = "./output/tchop35a/tchop35a.wav"
# out_path   = "./output/tchop35a/track0_audio.mp4"

# mux_video_audio(video_path, audio_path, out_path)
