# Inference script for test data
# Yu Sun (yu.sun@petermac.org)
#
#
# Run:
#  cd /workspace
#  python inference.py [data_dir] [output_dir]
#  python inference.py [data_dir] will use /results as output_dir
#
# E.g. python inference.py data_demo results_demo
#

import torch

from pathlib import Path
from data_loader import MyDataset

import numpy as np
from tqdm import tqdm

from nnunet_mednext import create_mednext_v1
import torch
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
code_dir = Path(__file__).parent

# Load model
model = create_mednext_v1(
    num_input_channels=7,
    num_classes=1,
    model_id="B",
    kernel_size=3,
    deep_supervision=False,
).to(device)

model.load_state_dict(torch.load(code_dir / "weights/7c_v2.pt"))


def inference(data_dir="/data", out_dir="/results", dose_div_factor=10):

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    cfig = {
        "train_bs": 4,
        "val_bs": 8,
        "csv_root": data_dir / "meta_data_test.csv",
        "scale_dose_dict": data_dir / "PTV_DICT.json",
        "pat_obj_dict": data_dir / "Pat_Obj_DICT.json",
        "num_workers": 4,
        "down_HU": -1000,
        "up_HU": 1000,
        "denom_norm_HU": 500,
        "in_size": [96, 128, 144],
        "out_size": [96, 128, 144],
        "norm_oar": True,
        "CatStructures": False,
        "dose_div_factor": 10,
    }

    ds = MyDataset(cfig, "test")

    # Inference: iterate through the dataset since aug is within
    for data_dict in tqdm(ds, "Running Inference"):
        # Get the first 7 channel (what the model was trained on)
        x = data_dict["data"].to(device)
        x = x[:-1].unsqueeze(0)

        # Get the prediction
        pred = model(x)[0] * dose_div_factor

        # Place back to original space
        pred_ori = ds.aug.inverse(dict(img=pred))["img"]
        pred_ori = pred_ori.detach().cpu().squeeze().numpy()

        # Save results
        outfile = out_dir / f'{data_dict["id"]}_pred.npy'
        np.save(outfile, pred_ori)
        print(f"[INFO] Results saved: {outfile}")
    print("Inference finished")


if __name__ == "__main__":
    print(__file__)
    if len(sys.argv) == 3:
        data_dir, out_dir = sys.argv[1:]
        inference(data_dir, out_dir)
    elif len(sys.argv) == 2:
        data_dir = sys.argv[1]
        inference(data_dir)
    else:
        print("Invalid input")
