import os
import argparse
import pandas as pd
import torch
import numpy as np
from inference.classifier import TabPFNClassifier
import json
from data_transform import * 
from tqdm import tqdm
from pandas.errors import DtypeWarning
import warnings
import yaml
def model_save_path(model_file):
    model_save = model_file
    if "tabpfn-v2-classifier.ckpt" in model_file:
        model_save = model_file
    else:
        model_save = model_file.replace('.cpkt', '-mfp.cpkt')
        model_ours = model_file
        model_default = 'tabpfn-v2-classifier.ckpt'
        try:
            with open(model_ours, 'rb') as f:
                model_m = torch.load(f)
            with open(model_default, 'rb') as f:
                model_d = torch.load(f)
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            print(f"请检查文件是否存在: {model_ours} 和 {model_default}")
            exit(1)
        my_stat = model_m[0]

        new_stat = {}
        for k, v in my_stat.items():
            if 'criterion' in k:
                continue
            new_stat[k.replace('module.','')] = v
        
        model_d['state_dict']=new_stat
        model_d['config']['emsize'] = model_m[2]['emsize']
        model_d['config']['nhead'] = model_m[2]['nhead']
        model_d['config']['nlayers'] = model_m[2]['nlayers']
        model_d['config']['nhid_factor'] = model_m[2]['nhid_factor']
        model_d['config']['feature_positional_embedding'] = model_m[2]['feature_positional_embedding']     
        model_d['config']['pre_norm'] = model_m[2]['use_pre_norm']    
        torch.save(model_d, open(model_save,'wb'))
    return model_save
    
preprocess_config_pfn_input = {
    'max_unique_values_for_categorical': 20,
    'impute_numerical_strategy': 'None',
    'impute_categorical_strategy': 'missing',
    'categorical_encoding_strategy': 'label',
    'numerical_scaling_strategy': 'min-max',
    'y_encoding_strategy': 'label'
}

preprocess_config_diff_input = {
    'max_unique_values_for_categorical': 20,
    'impute_numerical_strategy': 'zero',
    'impute_categorical_strategy': 'missing',
    'categorical_encoding_strategy': 'one-hot',
    'numerical_scaling_strategy': 'min-max',
    'y_encoding_strategy': 'one-hot',
    'y_max_classes': 10,
}

def embeddings(X_trans:np.ndarray, y_trans:np.ndarray, classifier:TabPFNClassifier, embedding_strategy:str, embedding_params:dict):
    '''
    returns:

    emb: np.ndarray of shape [num_embeddings, emb_dim]
    '''
    num_embeddings_min = embedding_params.get('num_embeddings_min', 1)
    num_embeddings_max = embedding_params.get('num_embeddings_max', 10)
    num_embeddings = np.random.randint(num_embeddings_min, num_embeddings_max+1)
    emb = []
    if embedding_strategy == 'small_batch_average':
        for _ in range(num_embeddings):
            batch_size_max = min(10000, embedding_params.get('batch_size_max_ratio', 0.5) * X_trans.shape[0])
            batch_size_min = embedding_params.get('batch_size_min_ratio', 0.5) * X_trans.shape[0]
            batch_size  =  np.random.randint(batch_size_min, batch_size_max+1)
            batch_indices = np.random.choice(X_trans.shape[0], size=batch_size, replace=False)
            X_batch = X_trans[batch_indices]
            y_batch = y_trans[batch_indices]
            classifier.fit(X_batch, y_batch)
            emb_i = classifier.get_embeddings(X_batch, data_source='train') #[n_estimators, batch_size, emb_dim]
            emb_i = np.mean(emb_i, axis=(0,1)) # [emb_dim,]
            emb.append(emb_i)
        emb = np.stack(emb, axis=0) # [num_embeddings, emb_dim]
    return emb

def check_valid_dataset(df:pd.DataFrame, dataset_filter_params:dict):
    if 'min_samples' in dataset_filter_params:
        if df.shape[0] < dataset_filter_params['min_samples']:
            return False
    if 'max_samples' in dataset_filter_params:
        if df.shape[0] > dataset_filter_params['max_samples']:
            return False
    if 'min_features' in dataset_filter_params:
        if df.shape[1]-1 < dataset_filter_params['min_features']:
            return False
    if 'max_features' in dataset_filter_params:
        if df.shape[1]-1 > dataset_filter_params['max_features']:
            return False
    return True

if __name__ == "__main__":
    benchmark_base_dir = "/mnt/public/cls/classifier_benchmarks"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        # config["save_dir"] = os.path.join(config["save_base_dir"], config["experiment_name"])

    benchmark_name = config.get('benchmark_name', 'benchmark_504')
    model_save = model_save_path(config.get('model_file', 'tabpfn-v2-classifier.ckpt'))
    save_dir = config.get('save_dir', '/mnt/public/hxy/diff_data')
    pad_to = config.get('pad_to', 1024)
    emb_strategy = config.get('embedding_strategy', 'average_over_all_samples')
    emb_params = config['embedding_params'] if 'embedding_params' in config else {}
    dataset_filter_params = config.get('dataset_filter_params', {})
    if os.path.exists(save_dir) and config.get('overwrite', True):
        import shutil
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    classifier = TabPFNClassifier(device='cuda:0', ignore_pretraining_limits=True, model_path=model_save)

    dataset_csv = pd.read_csv(os.path.join(benchmark_base_dir, benchmark_name, benchmark_name + '.csv'))
    for dataset_name in tqdm(dataset_csv['dataset name']):
        dataset_dir = os.path.join(benchmark_base_dir, benchmark_name, dataset_name)

        # load train and test csv
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", DtypeWarning)
                train_csv = pd.read_csv(os.path.join(dataset_dir, dataset_name + '_train.csv'))
                if any(issubclass(warn.category, DtypeWarning) for warn in w):
                    print(f"Skip {dataset_name} because dtype warning")
                    continue
        except Exception as e:
            print(f"Skip {dataset_name} because read error: {str(e)}")
            continue

        test_csv = pd.read_csv(os.path.join(dataset_dir, dataset_name + '_test.csv'))
        df = pd.concat([train_csv, test_csv], ignore_index=True)

        # filter dataset with too few or too many samples or features
        if not check_valid_dataset(df, dataset_filter_params):
            continue

        # generate embeddings
        try: 
            X_trans, y_trans, _, _ = transform_df(df.copy(), **preprocess_config_pfn_input)
            X_trans = np.asarray(X_trans, dtype=np.float32)
            y_trans = np.asarray(y_trans, dtype=np.int64)
            all_embeddings = embeddings(X_trans=X_trans, y_trans=y_trans, classifier=classifier, embedding_strategy=emb_strategy, embedding_params=emb_params)
        except Exception as e:
            print(f"Skip {dataset_name} because embedding error: {str(e)}")
            continue
        
        num_samples = min(df.shape[0], config.get('max_num_samples_per_dataset', 3000))
        df = df.sample(n=num_samples, random_state=42, replace=False).reset_index(drop=True)
        X_trans_diff, y_trans_diff, X_preprocessor, y_preprocessor = transform_df(df.copy(), **preprocess_config_diff_input)
        X_trans_diff_tensor = torch.tensor(np.asarray(X_trans_diff), dtype=torch.float32)
        y_trans_diff_tensor = torch.tensor(y_trans_diff, dtype=torch.int64)
        
        emb_indices = np.random.choice(all_embeddings.shape[0], size=X_trans_diff_tensor.shape[0], replace=True)
        agg_emb = all_embeddings[emb_indices]
        agg_emb_tensor = torch.tensor(agg_emb, dtype=torch.float32)

        num_numerical_features = X_preprocessor.num_numerical
        num_categories = X_preprocessor.num_categories

        if X_trans_diff_tensor.shape[1] > pad_to:
            print(f"Skip {dataset_name} because feature dimension")
            continue
        num_classes = y_preprocessor.num_classes_
        num_categories_expended = []
        for i in range(len(num_categories)):
            num_categories_expended += [num_categories[i]] * num_categories[i]
        save_path = os.path.join(save_dir, f"{benchmark_name}_{dataset_name}")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        torch.save(X_trans_diff_tensor, os.path.join(save_path, 'X.pt'))
        torch.save(y_trans_diff_tensor, os.path.join(save_path, 'y.pt'))
        torch.save(agg_emb_tensor, os.path.join(save_path, 'emb.pt'))
        json.dump({
            'num_numerical_features': num_numerical_features,
            'num_categories_expanded': num_categories_expended,
            'num_categories': num_categories,
            'num_classes': num_classes,
            'max_classes': y_preprocessor.max_classes,
        }, open(os.path.join(save_path, 'meta.json'), 'w'), indent=4)

        
            
        