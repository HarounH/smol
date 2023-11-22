import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
import matplotlib.pyplot as plt
from typing import Optional, Union
from tqdm import tqdm
import logging
from smol.utils import setup_logger  # also calls setup_logger
from smol.plotting import plot1d
from collections import defaultdict
import pandas as pd
import wandb
from contextlib import nullcontext


logger = logging.getLogger(__name__)


def run(
    n: int,  # dataset size
    # learning algorithm
    batch_size: Union[int, float],
    n_epochs: int,
    lr: float,  # constant.
    threshold_type: str,
    weight_decay: float = 0.0,
    bias: bool = False,
    group: str = "debug",
    run_idx: int = 0,
    device: str = "cuda",
):
    x = torch.randn(n, 1, dtype=torch.float32, device=device)
    threshold = _get_threshold(threshold_type=threshold_type, x=x)
    y = (x > threshold).to(dtype=torch.float32)  # should be on cuda already

    acceptable_threshold_range = (x[y == 0].max().cpu().item(), x[y == 1].min().cpu().item())
    logger.info(f"threshold: lower={acceptable_threshold_range[0]:.4f} actual={threshold:.4f} upper={acceptable_threshold_range[1]:.4f}")

    if isinstance(batch_size, float):
        dl_batch_size = int(batch_size * n)
    else:
        dl_batch_size = batch_size

    run = wandb.init(
        project="smol_one_fp32_neuron",
        group=group,
        config={
            "n": n,
            "bias": bias,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "lr": lr,
            "threshold_type": threshold_type,
            "threshold": threshold,
            "threshold_lower": acceptable_threshold_range[0],
            "threshold_upper": acceptable_threshold_range[1],
            "run_idx": run_idx,
            "device": device,
            "weight_decay": weight_decay,
        }
    )

    img = plot1d(x=x, y=y, threshold=threshold, return_np=True)
    images = wandb.Image(
        img,
        caption="Dataset Visualization"
    )
    wandb.log({"images/dataset": images})


    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset=dataset, batch_size=dl_batch_size, shuffle=True)

    classifier = nn.Linear(1, 1, bias=bias).to(device=device)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    wandb.watch(classifier, log_freq=1, log="all")
    classifier.train()
    step: int = 0
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        losses = []
        for batch_id, (batch_x, batch_y) in enumerate(dataloader):
            yhat = classifier(batch_x) + (1.0 if not bias else 0.0) # constant bias
            loss = bce(input=yhat, target=batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            losses.append(loss.cpu().item())
            with torch.no_grad():
                implicitly_estimated_threshold = -(1.0 if not bias else classifier.bias.cpu().item()) / classifier.weight.cpu().item()

            wandb.log({
                "implicitly_estimated_threshold": implicitly_estimated_threshold,
                "loss": losses[-1],
                "step": step,
                "bias": classifier.bias.cpu().item() if bias else 1.0,
                "weight": classifier.weight.cpu().item(),
            })
        pbar.set_description(f"[{epoch}] loss: {np.mean(losses):.4f}")
    run.finish()


def _get_threshold(threshold_type: str, x: torch.Tensor) -> float:
    min_ = x.min()
    max_ = x.max()
    if threshold_type.startswith("constant"):

        thr = float(threshold_type.split("_")[1])
        if thr >= min_ and thr <= max_:
            return thr
        raise RuntimeError("Unacceptable constant")

    elif threshold_type.startswith("randn"):
        scale = float(threshold_type.split("_")[1])
        thr = scale * float(torch.randn(1).item())
        while thr <= min_ or thr >= max_:
            thr = scale * float(torch.randn(1).item())
        return thr

    elif threshold_type == "median":
        thr = float(x.median().item())
        while thr <= min_ or thr >= max_:
            thr = float(x.median().item())
        return thr

    raise RuntimeError(f"Unexpected {threshold_type=}")


if __name__ == "__main__":
    group = "v15_sweep_w_decay"
    # get_batch_size = lambda n: 64
    batch_sizes = [0.01, 0.05, 0.25, 1.0]
    ns = [128, 2 * 8192]
    threshold_types = ["randn_0.25"] #, "constant_0", ]
    lrs = [1e-2, 1e-1, 0.5, 1, 1.5, 2.0]
    weight_decays = [1e-2, 1e-3, 1e-4, 0.1, 0.5, 1.0, 2.0]
    biases = [True]
    n_runs = 1
    failures = []
    for run_idx in range(n_runs):
        for n in ns:
            for threshold_type in threshold_types:
                for lr in lrs:
                    for bias in biases:
                        for batch_size in batch_sizes:
                            for weight_decay in weight_decays:
                                try:
                                    with nullcontext():
                                        run(
                                            n=n,
                                            batch_size=batch_size,
                                            n_epochs=10,  # dont need to limit, just take earlier results during plotting
                                            lr=float(lr),
                                            threshold_type=threshold_type,
                                            bias=bias,
                                            group=group,
                                            run_idx=run_idx,
                                            weight_decay=weight_decay,
                                        )
                                except:
                                    logger.error(f"Failed on {n=} {threshold_type=} {lr=} {weight_decay=}", exc_info=True)
                                    failures.append(dict(
                                        n=n,
                                        batch_size=batch_size,
                                        n_epochs=10,  # dont need to limit, just take earlier results during plotting
                                        lr=float(lr),
                                        threshold_type=threshold_type,
                                        bias=bias,
                                        group=group,
                                        run_idx=run_idx,
                                        weight_decay=weight_decay,
                                    ))
    for failure in failures:
        msg = ""
        for k, v in failure.items():
            msg += f"{k}: {v} | "
        logger.info(f"failure: {msg}")