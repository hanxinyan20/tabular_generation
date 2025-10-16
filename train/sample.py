import torch
from diff_scripts.model_tabddpm import GaussianMultinomialDiffusion
from .utils_train import get_model
import os
import numpy as np
import pandas as pd
from diff_scripts.data_transform import *
from diff_scripts.preprocess import embeddings, model_save_path
from inference.classifier import TabPFNClassifier
def load_model(
    ckpt_path,
    device,
    model_type:str,
    model_params = None,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    num_timesteps = 1000,
):
    
    model = get_model(
        model_type,
        model_params,
    )


    diffusion = GaussianMultinomialDiffusion(
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type, 
        num_timesteps=num_timesteps,
        scheduler=scheduler,
    )

    data = torch.load(ckpt_path, map_location=device, weights_only=True)
    diffusion.load_state_dict(data['model'])

    diffusion.to(device)
    diffusion.eval()
    return diffusion

def conditional_sample(
    ckpt_path,
    device,
    model_type:str,
    model_params:dict,
    gaussian_loss_type:str,
    scheduler:str,
    num_timesteps:int,
    num_samples:int,

    y: torch.Tensor, 
    emb: torch.Tensor,
    num_classes_expanded:list,
    num_numerical_features:int, 
    num_classes:list,
    sample_batch_size:int,
):
    diffusion = load_model(
        ckpt_path,
        device,
        model_type,
        model_params,
        gaussian_loss_type,
        scheduler,
        num_timesteps,
    )

    all_Xs = []
    for i in range(0, num_samples, sample_batch_size):
        curr_num_samples = min(sample_batch_size, num_samples-i)
        Xs = diffusion.sample(
            num_samples=curr_num_samples,
            y=y[i:i+curr_num_samples],
            emb=emb[i:i+curr_num_samples],
            num_classes_expanded=num_classes_expanded,
            num_numerical_features=num_numerical_features,
            num_classes=num_classes,
        )
        Xs = Xs.detach().cpu().numpy()
        all_Xs.append(Xs)

    all_Xs = np.concatenate(all_Xs, axis=0)

    return all_Xs

# preprocess_config_pfn_input = {
#     'max_unique_values_for_categorical': 20,
#     'impute_numerical_strategy': 'None',
#     'impute_categorical_strategy': 'missing',
#     'categorical_encoding_strategy': 'label',
#     'numerical_scaling_strategy': 'min-max',
#     'y_encoding_strategy': 'label'
# }

# preprocess_config_diff_input = {
#     'max_unique_values_for_categorical': 20,
#     'impute_numerical_strategy': 'zero',
#     'impute_categorical_strategy': 'missing',
#     'categorical_encoding_strategy': 'one-hot',
#     'numerical_scaling_strategy': 'min-max',
#     'y_encoding_strategy': 'one-hot',
#     'y_max_classes': 10,
# }

def expand_dataset(
    df: pd.DataFrame,
    preprocess_config_pfn_input: dict,
    preprocess_config_diff_input: dict,
    model_file:str,
    emb_strategy: str,
    embedding_params: dict,
    num_samples:int,

    ckpt_path:str,
    model_type:str,
    model_params:dict,
    gaussian_loss_type:str,
    scheduler:str,
    num_timesteps:int,
    sample_batch_size:int,
    emb_path:str = None,
):
    device = 'cuda:0'
    X_trans, y_trans, _, _ = transform_df(df.copy(), **preprocess_config_pfn_input)
    print(f"X_trans.columns: {X_trans.columns}")
    X_trans = np.asarray(X_trans, dtype=np.float32)
    y_trans = np.asarray(y_trans, dtype=np.int64) # shape [n, 10]
    embedding_params['batch_size_max_ratio'] = 1.0
    embedding_params['batch_size_min_ratio'] = 1.0
    embedding_params['num_embeddings_min'] = 1
    embedding_params['num_embeddings_max'] = 1
    model_save = model_save_path(model_file)
    
    if emb_path is not None and os.path.exists(emb_path):
        print(f"Loading precomputed embeddings from {emb_path}")
        emb = torch.load(emb_path).to(device) # [n, emb_dim]
        # 随机选择num_samples个embedding
        indices = np.random.choice(emb.size(0), size=num_samples, replace=True)
        emb = emb[indices]
        print(f"emb shape: {emb.shape}")
    else:
        classifier = TabPFNClassifier(device='cuda:0', ignore_pretraining_limits=True, model_path=model_save)
        emb = embeddings(X_trans, y_trans, classifier, emb_strategy, embedding_params)# np.ndarray of shape [1, emb_dim]
        emb = np.repeat(emb, num_samples, axis=0)
        emb = torch.from_numpy(emb).to(torch.float32).to(device)
    
    X_trans_diff, y_trans_diff, X_preprocessor, y_preprocessor = transform_df(df.copy(), **preprocess_config_diff_input)
    columns = X_trans_diff.columns.tolist()
    y_trans_diff_tensor = torch.tensor(y_trans_diff, dtype=torch.int64)
    idx = torch.randint(0, y_trans_diff_tensor.size(0), (num_samples,), device=y_trans_diff_tensor.device)  # 随机索引
    y_sampled = y_trans_diff_tensor[idx].to(device)

    num_numerical_features = X_preprocessor.num_numerical
    num_categories = X_preprocessor.num_categories
    num_categories_expended = []
    for i in range(len(num_categories)):
        num_categories_expended += [num_categories[i]] * num_categories[i]
    X_sampled = conditional_sample(
        ckpt_path=ckpt_path,
        device=device,
        model_type=model_type,
        model_params=model_params,
        gaussian_loss_type=gaussian_loss_type,
        scheduler=scheduler,
        num_timesteps=num_timesteps,
        num_samples=num_samples,
        y=y_sampled,
        emb=emb,
        num_classes_expanded=num_categories_expended,
        num_numerical_features=num_numerical_features,
        num_classes=num_categories,
        sample_batch_size=sample_batch_size,
    )
    X_sampled_array = np.array(X_sampled)
    # 把X_sampled_array的num_numerical_features之后的列变成0或1
    if len(num_categories) > 0:
        X_sampled_array[:, num_numerical_features:] = (X_sampled_array[:, num_numerical_features:] > 0.5).astype(np.float32)
    print(f"X_sampled_array before inverse transform: {X_sampled_array}")
    # 用X_preprocessor把X_sampled_array反变换回来
    # 把X_sampled_array变成DataFrame
    X_sampled_df = pd.DataFrame(X_sampled_array, columns=columns)
    X_sampled_df = X_preprocessor.inverse_transform(X_sampled_df)
    y_sampled = y_preprocessor.inverse_transform(y_sampled.detach().cpu().numpy())
    print(f"X_sampled_array after inverse transform: {X_sampled_array}")
    print(f"y_sampled after inverse transform: {y_sampled}")
    X_original_array = X_preprocessor.inverse_transform(X_trans_diff)
    print(f"X_original_array: {X_original_array[:2]}")
    return X_sampled_df, y_sampled
    






    



    



    # np.save(os.path.join(sample_save_dir, 'sampled_X.npy'), all_Xs)
    # np.save(os.path.join(sample_save_dir, 'sampled_y.npy'), all_ys)

    
    

