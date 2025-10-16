
from diff_scripts.train import expand_dataset
import argparse
import yaml
import os
import pandas as pd
def set_seed(seed:int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess_config', type=str, required=True, help='The sample config file with that used in preprocessing data.')
    parser.add_argument('--train_config', type=str, required=True, help='The train config file with that used in training the diffusion model.')
    parser.add_argument('--benchmark_dir', type=str, required=True, help='The benchmark base dir that contains all datasets.')
    parser.add_argument('--dataset_name', type=str, required=True, help='The name of the dataset to be processed.')
    parser.add_argument('--num_samples', type=int, default=2, help='Number of samples to generate.')
    parser.add_argument('--ckpt_path', type=str, required=True, help='The path to the trained diffusion model checkpoint.')
    parser.add_argument('--sample_save_dir', type=str, default='./expanded_data', help='The dir to save the expanded dataset.')
    parser.add_argument('--sample_batch_size', type=int, default=2, help='The batch size for sampling.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for sampling.')
    parser.add_argument('--emb_path', type=str, default=None, help='If specified, load precomputed embeddings from this path.')
    args = parser.parse_args()
    with open(args.preprocess_config, 'r') as f:
        preprocess_config = yaml.safe_load(f)
    with open(args.train_config, 'r') as f:
        train_config = yaml.safe_load(f)
    benchmark_dir = args.benchmark_dir
    dataset_name = args.dataset_name
    dataset_dir = f"{benchmark_dir}/{dataset_name}"
    train_csv = pd.read_csv(os.path.join(dataset_dir, dataset_name + '_train.csv'))
    test_csv = pd.read_csv(os.path.join(dataset_dir, dataset_name + '_test.csv'))
    df = pd.concat([train_csv, test_csv], ignore_index=True)

    set_seed(args.seed)
    X_sampled_df, y_sampled_array = expand_dataset(
        df=df,
        preprocess_config_pfn_input=preprocess_config['preprocess_config_pfn_input'],
        preprocess_config_diff_input=preprocess_config['preprocess_config_diff_input'],
        model_file=preprocess_config['model_file'],
        emb_strategy=preprocess_config['embedding_strategy'],
        embedding_params=preprocess_config['embedding_params'],
        num_samples=args.num_samples,
        ckpt_path=args.ckpt_path,
        model_type=train_config['model_type'],
        model_params=train_config['model_params'],
        gaussian_loss_type=train_config['gaussian_loss_type'],
        scheduler=train_config['scheduler'],
        num_timesteps=train_config['num_timesteps'],
        sample_batch_size=2,
        emb_path=args.emb_path, 
    )
    os.makedirs(args.sample_save_dir, exist_ok=True)
    # 把X 和 y 合并成一个DataFrame
    X_sampled_df['target'] = y_sampled_array
    X_sampled_df.to_csv(os.path.join(args.sample_save_dir, f"{dataset_name}_expanded_data.csv"), index=False)
    # also save the original data
    df.to_csv(os.path.join(args.sample_save_dir, f"{dataset_name}_original_data.csv"), index=False)

