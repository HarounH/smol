import torch
import matplotlib.pyplot as plt
from typing import Optional, Union, List
import numpy as np


LILAC_RGB = [200/255., 162/255., 200/255.]


def register_seaborn_palettes() -> None:
    # modifies seaborn color palettes registry in place
    # to add a new
    from seaborn.palettes import SEABORN_PALETTES, QUAL_PALETTE_SIZES
    SEABORN_PALETTES["cherry_blossoms"] = [
        # https://www.color-hex.com/color-palette/19094
        "#FF9FD1",
        "#E8E8E8",
        "#E2D298",
        "#8BE6F7",
        "#1FA9CC",
        # https://www.color-hex.com/color-palette/20985
        "#BAE4E5",
        "#6AC7C9",
        "#CE2C65",
        "#2D1700",
        "#ECECEC",
    ]
    QUAL_PALETTE_SIZES["cherry_blossoms"] = len(SEABORN_PALETTES["cherry_blossoms"])

@torch.no_grad()
def plot1d(x: torch.Tensor, y: torch.Tensor, threshold: Optional[Union[List[float], float]], return_np: bool = True) -> Optional[np.ndarray]:
    if threshold is None:
        threshold = []
    elif isinstance(threshold, float):
        threshold = [threshold]
    xc0 = x[y == 0].detach().clone().cpu()
    xc1 = x[y == 1].detach().clone().cpu()
    fig = plt.figure(figsize=(8, 1))
    canvas = fig.canvas
    ax = fig.gca()
    ax.scatter(xc0, torch.zeros_like(xc0), label='y=0', alpha=0.7)
    ax.scatter(xc1, torch.ones_like(xc1), label='y=1', alpha=0.7)
    for t in threshold:
        ax.axvline(x=t, color='r', linestyle='--', label=f'Threshold: {t}')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Label')
    fig.legend()
    plt.grid(True)

    # Remove y-axis ticks and tick labels
    plt.yticks([])

    if return_np:
        # https://stackoverflow.com/questions/35355930/figure-to-image-as-a-numpy-array
        canvas.draw()  # Draw the canvas, cache the renderer
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
        # NOTE: reversed converts (W, H) from get_width_height to (H, W)
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)
        return image
    plt.show()