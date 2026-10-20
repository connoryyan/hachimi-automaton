"""交互式 CLI：Midi 选择、轨道配置。"""

import glob
import os
import logging
from typing import Any

from hachimi.config import (
    MIDI_GLOB,
    AVAILABLE_CLASSES,
    DEFAULT_TRACK_CLASSES,
    DEFAULT_OCTAVE_SHIFT,
    DEFAULT_VOLUME_FACTOR,
)

logger = logging.getLogger(__name__)


def select_model_kind() -> str:
    """交互式选择歌词生成模型类型。

    Returns:
        ``"markov"`` 或 ``"transformer"``。
    """
    print("\n可选歌词生成模型：")
    print("  [0] markov      字符马尔科夫链（快、稳定）")
    print("  [1] transformer GPT-2 式神经网络（更连贯）")

    while True:
        choice = input("请选择模型编号 (默认 0 [markov]): ").strip()
        if choice == "":
            return "markov"
        if choice == "0":
            return "markov"
        if choice == "1":
            return "transformer"
        print("输入无效，请输入 0 或 1。")


def select_midi() -> str:
    """交互式选择 MIDI 文件。

    Returns:
        选中的 MIDI 文件路径。
    """
    midi_files = sorted(glob.glob(MIDI_GLOB, recursive=True))
    if not midi_files:
        print("未找到 MIDI 文件。")
        raise SystemExit(1)

    print("\n找到以下 MIDI 文件：")
    for i, path in enumerate(midi_files):
        print(f"  [{i}] {path}")

    while True:
        try:
            choice = int(input("\n请输入要处理的 MIDI 文件编号: "))
            if 0 <= choice < len(midi_files):
                return midi_files[choice]
        except ValueError:
            pass
        print("输入无效，请输入正确的数字。")


def configure_tracks(
    n_tracks: int,
    instrument_names: list[str],
    is_drum_flags: list[bool],
) -> tuple[list[str], list[int], list[float]]:
    """交互式配置每个 MIDI 轨道的角色、八度偏移、音量。

    Args:
        n_tracks: 轨道数。
        instrument_names: 每个轨道的名称。
        is_drum_flags: 每个轨道是否是打击乐。

    Returns:
        (track_classes, octave_shift, volume_factor) 三个列表。
    """
    track_classes: list[str] = []
    octave_shift: list[int] = []
    volume_factor: list[float] = []

    print("\n可选角色:")
    for i, cls in enumerate(AVAILABLE_CLASSES):
        print(f"  [{i}] {cls}")

    print("\n--- 请配置各轨道参数 (直接回车使用默认值) ---")
    for i in range(n_tracks):
        name = instrument_names[i] or "未命名轨道"
        is_drum = is_drum_flags[i]

        def_cls_idx = DEFAULT_TRACK_CLASSES[i] if i < len(DEFAULT_TRACK_CLASSES) else 0
        default_cls = AVAILABLE_CLASSES[def_cls_idx]
        default_shift = (
            DEFAULT_OCTAVE_SHIFT[i] if i < len(DEFAULT_OCTAVE_SHIFT) else 0
        )
        default_vol = (
            DEFAULT_VOLUME_FACTOR[i] if i < len(DEFAULT_VOLUME_FACTOR) else 1.0
        )

        if is_drum:
            print(f"\n轨道 [{i}] {name}: 检测为打击乐，默认静音。")
            track_classes.append(AVAILABLE_CLASSES[0])
            octave_shift.append(0)
            volume_factor.append(0.0)
            continue

        print(f"\n配置轨道 [{i}] {name}:")

        # 角色
        cls_input = input(
            f"  选择角色编号 (0-{len(AVAILABLE_CLASSES)-1}, "
            f"默认 {def_cls_idx} [{default_cls}]): "
        )
        if cls_input.strip() == "":
            track_classes.append(default_cls)
        else:
            try:
                idx = int(cls_input)
                track_classes.append(AVAILABLE_CLASSES[idx] if 0 <= idx < len(AVAILABLE_CLASSES) else default_cls)
            except ValueError:
                track_classes.append(default_cls)

        # 八度
        shift_input = input(f"  八度偏移 (例如 -1, 0, 1, 默认 {default_shift}): ")
        if shift_input.strip() == "":
            octave_shift.append(default_shift)
        else:
            try:
                octave_shift.append(int(shift_input))
            except ValueError:
                octave_shift.append(default_shift)

        # 音量
        vol_input = input(f"  音量比例 (0.0-1.0, 默认 {default_vol}): ")
        if vol_input.strip() == "":
            volume_factor.append(default_vol)
        else:
            try:
                volume_factor.append(float(vol_input))
            except ValueError:
                volume_factor.append(default_vol)

    return track_classes, octave_shift, volume_factor
