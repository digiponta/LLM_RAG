"""Small, explicitly authored Japanese definition-QA probe for v0.9.7.

No Real Persona Chat content, no automatic web claims, no hidden labeling.
Train/validation/test subjects are disjoint, so test evaluates transfer to
unseen concepts rather than memorizing exact definitions. This is a small
development probe, NOT a rigorous factuality benchmark.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

# All examples are authored for this experiment, not copied from user chats.
CONCEPTS = {
    "train": [
        ("力", "物体の運動状態を変えたり変形させたりする作用です。"),
        ("質量", "物体の慣性の大きさを表す物理量です。"),
        ("速度", "位置の時間変化を表す量です。"),
        ("加速度", "速度が時間とともに変わる割合です。"),
        ("温度", "物体の熱的な状態を表す物理量です。"),
        ("エネルギー", "仕事や熱として移り変わる保存量です。"),
        ("電流", "電荷が流れる割合を表す量です。"),
        ("電圧", "二点間の電位の差です。"),
        ("光", "人の目で見える範囲の電磁波です。"),
        ("音", "媒質中を伝わる圧力などの振動です。"),
        ("波長", "周期的な波の隣り合う山の間隔です。"),
        ("周波数", "一秒間の振動回数を表す量です。"),
        ("原子", "原子核と電子からなる物質の基本的な単位です。"),
        ("分子", "原子が結合してできた粒子です。"),
        ("電子", "負の電荷を持つ素粒子です。"),
        ("陽子", "正の電荷を持ち原子核を構成する粒子です。"),
        ("重力", "質量やエネルギーに関係して生じる相互作用です。"),
        ("惑星", "恒星などの周囲を公転する天体の一種です。"),
        ("恒星", "核融合などによって自ら光を放つ天体です。"),
        ("銀河", "多数の恒星やガスなどが重力で集まった天体系です。"),
    ],
    "validation": [
        ("電磁波", "電場と磁場の振動として伝わる波です。"),
        ("磁場", "磁石や電流などに関係する物理的な場です。"),
        ("密度", "単位体積あたりの質量です。"),
        ("圧力", "単位面積あたりに働く垂直方向の力です。"),
        ("熱", "温度差などによって移動するエネルギーです。"),
        ("衛星", "惑星などの周囲を公転する天体です。"),
    ],
    "test": [
        ("宇宙", "空間と時間、そこに存在する物質やエネルギーの全体です。"),
        ("時間", "出来事の順序や持続を記述するための物理量です。"),
        ("空間", "物体の位置や距離を表現するための広がりです。"),
        ("真空", "物質がほとんど存在しない状態の空間です。"),
        ("量子", "物理量の離散的な単位や量子状態を表す概念です。"),
        ("波動関数", "量子状態を表現する数学的な関数です。"),
        ("相対性理論", "時間や空間、重力を扱うアインシュタインの理論です。"),
        ("ブラックホール", "光も脱出できない領域を持つ時空の構造です。"),
    ],
}


def build_rows(split: str):
    if split not in CONCEPTS:
        raise ValueError("Unknown split")
    return [
        {"id": f"concept-v097:{split}:{i}",
         "conversation_id": f"concept-v097:{split}:{i}",
         "question": f"{term}とは",
         "answer": definition,
         "context": "",
         "example_type": "hand_authored_definition_probe"}
        for i, (term, definition) in enumerate(CONCEPTS[split])
    ]


def run(args):
    terms = {split: {x[0] for x in values} for split, values in CONCEPTS.items()}
    for i, first in enumerate(terms):
        for second in list(terms)[i+1:]:
            if terms[first] & terms[second]:
                raise ValueError(f"Concept train/validation/test overlap: {first}, {second}")
    paths = {name: args.output_dir / f"concept_{name}_v097.jsonl"
             for name in CONCEPTS}
    if not args.overwrite:
        for file in paths.values():
            if file.exists():
                raise FileExistsError(f"Existing file: {file}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        with path.open("w", encoding="utf-8") as fh:
            for row in build_rows(name):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"counts": {name: len(values) for name, values in CONCEPTS.items()},
                      "output_dir": str(args.output_dir),
                      "warning": "Tiny authored definitions: useful only as a development probe, not a broad factuality benchmark."},
                     ensure_ascii=False, indent=2))
    return paths


def main():
    p = argparse.ArgumentParser(description="Generate tiny public-data definition-QA probe")
    p.add_argument("--output-dir", type=Path, default=Path("data/dialogue/concepts_v097"))
    p.add_argument("--overwrite", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
