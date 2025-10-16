import numpy as np
import os
import lib
from diff_scripts.model_tabddpm.modules import MLPDiffusion, ResNetDiffusion
from sklearn.model_selection import train_test_split
import pickle
def get_model(
    model_name,
    model_params,
): 
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    elif model_name == 'resnet':
        model = ResNetDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)

def get_task_type_and_num_classes(dataset_id, cache_dir):
    df, categorical_indicators, target_attribute, _ = lib.load_openml_dataset(dataset_id, cache_dir)
    columns = list(df.columns)
    
    if target_attribute not in columns:
        raise ValueError(f"Target attribute {target_attribute} not in columns {columns}")

    categorical_cols = [c for i, c in enumerate(columns) if categorical_indicators[i]]
    numeric_cols = [c for i, c in enumerate(columns) if not categorical_indicators[i]]
    num_classes = 0
    if target_attribute in categorical_cols:
        categorical_cols.remove(target_attribute)
        num_classes = df[target_attribute].nunique()
        task_type = 'binclass' if num_classes == 2 else 'multiclass'
    if target_attribute in numeric_cols:
        numeric_cols.remove(target_attribute)
        task_type = 'regression'
    return lib.TaskType(task_type), num_classes

def make_synthetic_dataset(
    data_path: str,
    T: lib.Transformations,
    split_seed: int = 0,
):
    assert data_path.endswith('.pkl'), "Only support .pkl files for synthetic datasets."
    # 只支持分类任务
    dataset_name = data_path.split('/')[-1].replace('.pkl', '')
    dataset_dir = data_path.replace('.pkl', '')
    data = pickle.load(open(data_path, 'rb'))
    X = data['x'].squeeze(1)
    y_np = data['y'].squeeze(1)

    X_num = []
    X_cat = []
    for i in range(X.shape[1]):
        if len(np.unique(X[:, i])) < 20 :
            X_cat.append(X[:, i].reshape(-1, 1))
        else:
            X_num.append(X[:, i].reshape(-1, 1))
    X_num_np = np.concatenate(X_num, axis=1) if len(X_num) > 0 else np.empty((X.shape[0], 0))
    X_cat_np = np.concatenate(X_cat, axis=1) if len(X_cat) > 0 else np.empty((X.shape[0], 0))
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_num_np, X_cat_np, y_np, test_size=0.2, random_state=split_seed
    )
    X_num_test, X_num_val, X_cat_test, X_cat_val, y_test, y_val = train_test_split(
        X_num_test, X_cat_test, y_test, test_size=0.5, random_state=split_seed
    )
    X_num = {
        'train': X_num_train,
        'val': X_num_val,
        'test': X_num_test
    } if X_num_np.shape[1] > 0 else None
    X_cat = {
        'train': X_cat_train,
        'val': X_cat_val,
        'test': X_cat_test
    } if X_cat_np.shape[1] > 0 else None
    y = {
        'train': y_train,
        'val': y_val,
        'test': y_test,
    }
    
    if os.path.exists(dataset_dir):
        os.system(f'rm -r {dataset_dir}')
    os.makedirs(dataset_dir)
    np.save(os.path.join(dataset_dir, 'X_num_train.npy'), X_num['train'] if X_num is not None else np.empty((X_cat['train'].shape[0], 0)))
    np.save(os.path.join(dataset_dir, 'X_num_val.npy'), X_num['val'] if X_num is not None else np.empty((X_cat['val'].shape[0], 0)))
    np.save(os.path.join(dataset_dir, 'X_num_test.npy'), X_num['test'] if X_num is not None else np.empty((X_cat['test'].shape[0], 0)))
    np.save(os.path.join(dataset_dir, 'X_cat_train.npy'), X_cat['train'] if X_cat is not None else np.empty((X_num['train'].shape[0], 0)))
    np.save(os.path.join(dataset_dir, 'X_cat_val.npy'), X_cat['val'] if X_cat is not None else np.empty((X_num['val'].shape[0], 0)))
    np.save(os.path.join(dataset_dir, 'X_cat_test.npy'), X_cat['test'] if X_cat is not None else np.empty((X_num['test'].shape[0], 0)))
    np.save(os.path.join(dataset_dir, 'y_train.npy'), y['train'])
    np.save(os.path.join(dataset_dir, 'y_val.npy'), y['val'])
    np.save(os.path.join(dataset_dir, 'y_test.npy'), y['test'])
    X_cat = {k: concat_y_to_X(X_cat[k], y[k]) for k in ['train', 'val', 'test']}  if X_cat is not None else {k: concat_y_to_X(None, y[k]) for k in ['train', 'val', 'test']}
    num_classes = len(np.unique(y_np))
    print("np.unique(y_np):", np.unique(y_np))
    task_type = 'binclass' if num_classes == 2 else 'multiclass'
    print("num_classes:", num_classes, "task_type:", task_type)
    print("X_num.shape:", X_num['train'].shape if X_num is not None else None)
    print("X_cat.shape:", X_cat['train'].shape if X_cat is not None else None)
    D = lib.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=lib.TaskType(task_type),
        n_classes=num_classes
    )
    return lib.transform_dataset(D, T, None)
    

def make_openml_dataset(
    dataset_id: int,
    cache_dir: str,
    T: lib.Transformations,
    change_val: bool = False,
    is_y_cond: bool = False,
    split_seed: int = 0,
    load_categorical_indicator: bool = True,
    detect_target_if_missing: bool = True,
):
    assert is_y_cond == False, "We just fit the joint distribution p(x, y) instead of p(x|y) for now."

    df, categorical_indicators, target_attribute, _ = lib.load_openml_dataset(dataset_id, cache_dir, load_categorical_indicator=load_categorical_indicator, detect_target_if_missing=detect_target_if_missing)
    columns = list(df.columns)
    if target_attribute not in columns:
        raise ValueError(f"Target attribute {target_attribute} not in columns {columns}")

    categorical_cols = [c for i, c in enumerate(columns) if categorical_indicators[i]]
    numeric_cols = [c for i, c in enumerate(columns) if not categorical_indicators[i]]
    num_classes = 0
    if target_attribute in categorical_cols:
        categorical_cols.remove(target_attribute)
        num_classes = df[target_attribute].nunique()
        task_type = 'binclass' if num_classes == 2 else 'multiclass'
    if target_attribute in numeric_cols:
        numeric_cols.remove(target_attribute)
        task_type = 'regression'

    X_cat_df = df[categorical_cols].dropna(axis=1, how='all')
    X_cat_np = X_cat_df.to_numpy()
   
    
    X_num_np = df[numeric_cols].to_numpy()
    y_np = df[target_attribute].to_numpy()
    if len(y_np.shape) == 1:
        y_np = y_np.reshape(-1, 1)
        
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_num_np, X_cat_np, y_np, test_size=0.2, random_state=split_seed
    )
    X_num_test, X_num_val, X_cat_test, X_cat_val, y_test, y_val = train_test_split(
        X_num_test, X_cat_test, y_test, test_size=0.5, random_state=split_seed
    )
    print("X_num_np.shape:", X_num_np.shape if X_num_np is not None else None)
    X_num = {
        'train': X_num_train,
        'val': X_num_val,
        'test': X_num_test
    } if X_num_np.shape[1] > 0 else None
    X_cat = {
        'train': X_cat_train, 
        'val': X_cat_val ,
        'test': X_cat_test
    } if X_cat_np.shape[1] > 0 else None
    y = {
        'train': y_train,
        'val': y_val,
        'test': y_test,
    } if len(y_np.shape) > 0 else None
    
    real_dataset_dir = os.path.join(cache_dir, f'{dataset_id}', 'real_data')
    if os.path.exists(real_dataset_dir):
        os.system(f'rm -r {real_dataset_dir}')

        
    os.makedirs(real_dataset_dir)
    np.save(os.path.join(real_dataset_dir, 'X_num_train.npy'), X_num['train'] if X_num is not None else np.empty((X_cat['train'].shape[0], 0)))
    np.save(os.path.join(real_dataset_dir, 'X_num_val.npy'), X_num['val'] if X_num is not None else np.empty((X_cat['val'].shape[0], 0)))
    np.save(os.path.join(real_dataset_dir, 'X_num_test.npy'), X_num['test'] if X_num is not None else np.empty((X_cat['test'].shape[0], 0)))
    np.save(os.path.join(real_dataset_dir, 'X_cat_train.npy'), X_cat['train'] if X_cat is not None else np.empty((X_num['train'].shape[0], 0)))
    np.save(os.path.join(real_dataset_dir, 'X_cat_val.npy'), X_cat['val'] if X_cat is not None else np.empty((X_num['val'].shape[0], 0)))
    np.save(os.path.join(real_dataset_dir, 'X_cat_test.npy'), X_cat['test'] if X_cat is not None else np.empty((X_num['test'].shape[0], 0)))
    np.save(os.path.join(real_dataset_dir, 'y_train.npy'), y['train'])
    np.save(os.path.join(real_dataset_dir, 'y_val.npy'), y['val'])
    np.save(os.path.join(real_dataset_dir, 'y_test.npy'), y['test'])
        
        
    
    if num_classes > 0 and is_y_cond == False:
        X_cat = {k: concat_y_to_X(X_cat[k], y[k]) for k in ['train', 'val', 'test']}  if X_cat is not None else {k: concat_y_to_X(None, y[k]) for k in ['train', 'val', 'test']}
    if num_classes == 0 and is_y_cond == False:
        X_num = {k: concat_y_to_X(X_num[k], y[k]) for k in ['train', 'val', 'test']}  if X_num is not None else {k: concat_y_to_X(None, y[k]) for k in ['train', 'val', 'test']}
    

    D = lib.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=lib.TaskType(task_type),
        n_classes=num_classes if num_classes > 0 else None
    )
    
    if change_val:
        D = lib.change_val(D)
    
    return lib.transform_dataset(D, T, None)
    
def make_dataset(
    data_path: str,
    T: lib.Transformations,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool
):
    # classification
    if num_classes > 0:
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) or not is_y_cond else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} 

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = lib.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if not is_y_cond:
                X_cat_t = concat_y_to_X(X_cat_t, y_t)
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) or not is_y_cond else None
        y = {}

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = lib.read_pure_data(data_path, split)
            if not is_y_cond:
                X_num_t = concat_y_to_X(X_num_t, y_t)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t

    info = lib.load_json(os.path.join(data_path, 'info.json'))
    
    
    # print(X_num['train'].shape if X_num is not None else None) # (6400, 7)
    # print(X_cat['train'].shape if X_cat is not None else None) # (6400, 4)
    # print(X_cat['train'][0]) 
    # print(y['train'].shape) # (6400,)
    # num_classes: 2 is_y_cond: True n_classes None 
    # print("num_classes:", num_classes, "is_y_cond:", is_y_cond, "n_classes", info.get('n_classes'))
    D = lib.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=lib.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = lib.change_val(D)
    
    return lib.transform_dataset(D, T, None)