from pathlib import Path
import numpy as np

# 定数の定義
LABEL_COLORS = {
    0: (254, 195, 195),  # white
    1: (204, 66, 38),    # lugol 
    2: (57, 103, 177),   # indigo
    3: (96, 165, 53),    # nbi
    4: (86, 65, 72),     # outside
    5: (159, 190, 183),  # bucket
}
DEFAULT_COLOR = (148, 148, 148)  # gray

def visualize_label_timeline(labels: np.ndarray, output_dir: Path, video_name: str, save_name: str) -> None:
    """ラベルのタイムラインをSVG形式で可視化する"""
    # 主クラス（0-5）のみを抽出
    labels = labels.tolist()
    n_images = len(labels)
    
    # SVGの寸法を設定
    timeline_width = n_images
    timeline_height = n_images // 10
    
    # SVGファイルの作成
    svg_content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{timeline_width}" height="{timeline_height}" viewBox="0 0 {timeline_width} {timeline_height}">'
    ]
    
    # 各ラベルに対応する矩形を追加
    segment_width = timeline_width / n_images
    for i in range(n_images):
        label = labels[i]
        x1 = i * segment_width
        x2 = (i + 1) * segment_width
        width = x2 - x1
        
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)
        rgb = f"rgb({color[0]}, {color[1]}, {color[2]})"
        
        svg_content.append(f'<rect x="{x1}" y="0" width="{width}" height="{timeline_height}" fill="{rgb}" />')
    
    # SVGを閉じる
    svg_content.append('</svg>')
    
    # 保存パスを正しく設定
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / f'{save_name}_{video_name}.svg'
    
    # SVGファイルを保存
    with open(save_file, 'w') as f:
        f.write('\n'.join(svg_content))
    
    print(f'{save_name} timeline image saved at {save_file}')