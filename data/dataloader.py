
import torch
import os
from torch.utils.data import DataLoader, DistributedSampler
import json
from tqdm import tqdm
class VariableLengthDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir:str="/mnt/public/hxy/diff_data/", batch_size_per_gpu:int=2):
        self.data_list = []
        for dataset_name in tqdm(os.listdir(base_dir),desc="Loading datasets"):
            dataset_path = os.path.join(base_dir, dataset_name)
            if os.path.isdir(dataset_path):
                try:
                    X = torch.load(os.path.join(dataset_path, 'X.pt'))
                    y = torch.load(os.path.join(dataset_path, 'y.pt'))
                    emb = torch.load(os.path.join(dataset_path, 'emb.pt'))
                    meta = json.load(open(os.path.join(dataset_path, 'meta.json'), 'r'))
                    n_samples = X.shape[0]
                    for start_idx in range(0, n_samples, batch_size_per_gpu):
                        end_idx = min(start_idx + batch_size_per_gpu, n_samples)
                        batch_X = X[start_idx:end_idx]
                        batch_y = y[start_idx:end_idx]
                        batch_emb = emb[start_idx:end_idx]
                        assert batch_X.shape[0] == batch_emb.shape[0], f"{batch_X.shape}, {batch_emb.shape}"
                        self.data_list.append({'X': batch_X,  'y': batch_y, 'emb': batch_emb, 'meta': meta})
                except Exception as e:
                    print(f"Error loading dataset {dataset_name}: {e}")
        print(f"Loaded {len(self.data_list)} datasets from {base_dir}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def passthrough_collate(batch):
    assert len(batch) == 1
    item = batch[0]

    N = item['X'].shape[0]
    perm = torch.randperm(N)

    shuffled = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == N:
            shuffled[k] = v[perm]
        else:
            shuffled[k] = v 
    return shuffled


def get_dataloader_ddp(base_dir="/mnt/public/hxy/diff_data/", batch_size_per_gpu=2, num_workers=4, seed=42):
    dataset = VariableLengthDataset(base_dir=base_dir, batch_size_per_gpu=batch_size_per_gpu)

    sampler = DistributedSampler(
        dataset,
        shuffle=True,   
        seed=seed,
        drop_last=False
    )

    loader = DataLoader(
        dataset,
        batch_size=1,     
        sampler=sampler,     
        shuffle=False,        
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=passthrough_collate,
    )
    return loader
