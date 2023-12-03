import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
from smol.utils import setup_logger  # also calls setup_logger.
from typing import List, Tuple
import itertools
import logging
import argparse
from smol.bitt import bitt2bool, bitt_stringify, NAND, Foldl


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dtype", choices=["fp32", "bitt"])
    parser.add_argument("-nv", "--n_variables", type=int, default=3)
    return parser

class RandomSamplingTensorDataset:
    def __init__(self, n_variables: int = 2, batch_size: int = -1):
        values = [0, 1]
        data_x = torch.as_tensor(list(itertools.product(*[values for _ in range(n_variables)]))).to(torch.int8)

        # labels are AND of variables.
        # TODO: consider other options?
        data_y = data_x[:, 0].clone()
        for i in range(1, n_variables):
            data_y = data_y & data_x[:, i]

        self.tensors = [data_x, data_y]
        self.expected_n = self.tensors[0].shape[0]
        assert batch_size > 0
        self.batch_size = batch_size

    def __next__(self) -> List[torch.Tensor]:
        indices = torch.randint(low=0, high=self.expected_n, size=(self.batch_size,))
        return [t[indices, ...] for t in self.tensors]


def fp32_stuff(dataset, n_variables: int) -> None:
    fp32_net = nn.Sequential(nn.Linear(n_variables, 1))
    fp32_optimizer = torch.optim.SGD(fp32_net.parameters(), lr=0.5)
    criterion = nn.BCEWithLogitsLoss()
    for step in range(500):
        fp32_optimizer.zero_grad()
        batch_x, batch_y = next(dataset)
        yhat = fp32_net(batch_x.float())[:, 0]  # Only 1 dim
        loss = criterion(yhat, batch_y.float())
        loss.backward()
        fp32_optimizer.step()
        if step % 100 == 0:
            logging.info(f"{step}: {loss.item():.3f}")

    data_x, data_y = dataset.tensors
    logging.debug(f"data_x={data_x}")
    logging.debug(f"data_y={data_y}")
    logging.debug(f"yhat_indicator={(fp32_net(data_x.float()) > 0).to(torch.int8).view(-1)}")
    logging.debug(f"weight={fp32_net[0].weight}")
    logging.debug(f"bias={fp32_net[0].bias}")


class BitIndex(nn.Module):
    def __init__(self, data_size: Tuple[int], input_dtype = torch.int8, output_type = torch.int8) -> None:
        super().__init__()
        # round num_output to closest available dtype
        self.data_size = data_size
        self.input_dtype = input_dtype
        # Mapping from `data_size * input_dtype`` bits to `output_dtype`` bits
        self.output_dtype = output_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: bitwise_and does a broadcast
        raise NotImplementedError()

if __name__ == "__main__":
    args = get_parser().parse_args()
    n_variables = args.n_variables
    logging.info(f"{n_variables=}")
    dataset = RandomSamplingTensorDataset(n_variables=n_variables, batch_size=2)

    if args.dtype == "fp32":
        fp32_stuff(dataset=dataset, n_variables=n_variables)
        exit()

    assert args.dtype == "bitt"
    # bitt stuff
    inn = nn.Sequential(
        NAND(data_size=(n_variables,), dtype=torch.int8),
        Foldl(f=torch.bitwise_xor, dtype=torch.int8),
    )
    data_x, data_y = dataset.tensors
    yhat = inn(data_x)

    # TODO: insert learning algorithm!

    logging.debug(f"data_x={data_x}")
    logging.debug(f"data_y={data_y}")
    logging.debug(f"yhat={yhat}")
    logging.debug(f"inn.weight={bitt_stringify(inn[0].weight)}")
