import argparse
import atexit
import enum
import json
import os
import pickle
import shutil
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, List, Dict, Type, Optional, Tuple, TypeVar, Union, cast, get_args, get_origin

import __main__
import numpy as np
import tomli
import tomli_w
import torch


from . import env

RawConfig = Dict[str, Any]
Report = Dict[str, Any]
T = TypeVar('T')


class Part(enum.Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    def __str__(self) -> str:
        return self.value


class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value



def update_training_log(training_log, data, metrics):
    def _update(log_part, data_part):
        for k, v in data_part.items():
            if isinstance(v, dict):
                _update(log_part.setdefault(k, {}), v)
            elif isinstance(v, list):
                log_part.setdefault(k, []).extend(v)
            else:
                log_part.setdefault(k, []).append(v)

    _update(training_log, data)
    transposed_metrics = {}
    for part, part_metrics in metrics.items():
        for metric_name, value in part_metrics.items():
            transposed_metrics.setdefault(metric_name, {})[part] = value
    _update(training_log, transposed_metrics)


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')


def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)


_CONFIG_NONE = '__none__'


def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def pack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x is None, _CONFIG_NONE))
    return config


def load_config(path: Union[Path, str]) -> Any:
    with open(path, 'rb') as f:
        return unpack_config(tomli.load(f))


def dump_config(config: Any, path: Union[Path, str]) -> None:
    with open(path, 'wb') as f:
        tomli_w.dump(pack_config(config), f)
    # check that there are no bugs in all these "pack/unpack" things
    assert config == load_config(path)


def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    Path(path).write_bytes(pickle.dumps(x, **kwargs))


def load(path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'load_{Path(path).suffix[1:]}'](Path(path), **kwargs)


def dump(x: Any, path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'dump_{Path(path).suffix[1:]}'](x, Path(path), **kwargs)


def _get_output_item_path(
    path: Union[str, Path], filename: str, must_exist: bool
) -> Path:
    path = env.get_path(path)
    if path.suffix == '.toml':
        path = path.with_suffix('')
    if path.is_dir():
        path = path / filename
    else:
        assert path.name == filename
    assert path.parent.exists()
    if must_exist:
        assert path.exists()
    return path


def load_report(path: Path) -> Report:
    return load_json(_get_output_item_path(path, 'report.json', True))


def dump_report(report: dict, path: Path) -> None:
    dump_json(report, _get_output_item_path(path, 'report.json', False))


def load_predictions(path: Path) -> Dict[str, np.ndarray]:
    with np.load(_get_output_item_path(path, 'predictions.npz', True)) as predictions:
        return {x: predictions[x] for x in predictions}


def dump_predictions(predictions: Dict[str, np.ndarray], path: Path) -> None:
    np.savez(_get_output_item_path(path, 'predictions.npz', False), **predictions)


def dump_metrics(metrics: Dict[str, Any], path: Path) -> None:
    dump_json(metrics, _get_output_item_path(path, 'metrics.json', False))


def load_checkpoint(path: Path, *args, **kwargs) -> Dict[str, np.ndarray]:
    return torch.load(
        _get_output_item_path(path, 'checkpoint.pt', True), *args, **kwargs
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        assert os.environ.get('CUDA_VISIBLE_DEVICES') is not None
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def _print_sep(c, size=100):
    print(c * size)



_LAST_SNAPSHOT_TIME = None


def backup_output(output_dir: Path) -> None:
    backup_dir = os.environ.get('TMP_OUTPUT_PATH')
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if backup_dir is None:
        assert snapshot_dir is None
        return
    assert snapshot_dir is not None

    try:
        relative_output_dir = output_dir.relative_to(env.PROJ)
    except ValueError:
        return

    for dir_ in [backup_dir, snapshot_dir]:
        new_output_dir = dir_ / relative_output_dir
        prev_backup_output_dir = new_output_dir.with_name(new_output_dir.name + '_prev')
        new_output_dir.parent.mkdir(exist_ok=True, parents=True)
        if new_output_dir.exists():
            new_output_dir.rename(prev_backup_output_dir)
        shutil.copytree(output_dir, new_output_dir)
        # the case for evaluate.py which automatically creates configs
        if output_dir.with_suffix('.toml').exists():
            shutil.copyfile(
                output_dir.with_suffix('.toml'), new_output_dir.with_suffix('.toml')
            )
        if prev_backup_output_dir.exists():
            shutil.rmtree(prev_backup_output_dir)

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        import nirvana_dl.snapshot  # type: ignore[code]

        nirvana_dl.snapshot.dump_snapshot()
        _LAST_SNAPSHOT_TIME = time.time()
        print('The snapshot was saved!')


def _get_scores(metrics: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, float]]:
    return (
        {k: v['score'] for k, v in metrics.items()}
        if 'score' in next(iter(metrics.values()))
        else None
    )


def format_scores(metrics: Dict[str, Dict[str, Any]]) -> str:
    return ' '.join(
        f"[{x}] {metrics[x]['score']:.3f}"
        for x in ['test', 'val', 'train']
        if x in metrics
    )


def finish(output_dir: Path, report: dict) -> None:
    print()
    _print_sep('=')

    metrics = report.get('metrics')
    if metrics is not None:
        scores = _get_scores(metrics)
        if scores is not None:
            dump_json(scores, output_dir / 'scores.json')
            print(format_scores(metrics))
            _print_sep('-')

    dump_report(report, output_dir)
    json_output_path = os.environ.get('JSON_OUTPUT_FILE')
    if json_output_path:
        try:
            key = str(output_dir.relative_to(env.PROJ))
        except ValueError:
            pass
        else:
            json_output_path = Path(json_output_path)
            try:
                json_data = json.loads(json_output_path.read_text())
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                json_data = {}
            json_data[key] = load_json(output_dir / 'report.json')
            json_output_path.write_text(json.dumps(json_data, indent=4))
        shutil.copyfile(
            json_output_path,
            os.path.join(os.environ['SNAPSHOT_PATH'], 'json_output.json'),
        )

    output_dir.joinpath('DONE').touch()
    backup_output(output_dir)
    print(f'Done! | {report.get("time")} | {output_dir}')
    _print_sep('=')
    print()


def from_dict(datacls: Type[T], data: dict) -> T:
    assert is_dataclass(datacls)
    data = deepcopy(data)
    for field in fields(datacls):
        if field.name not in data:
            continue
        if is_dataclass(field.type):
            data[field.name] = from_dict(field.type, data[field.name])
        elif (
            get_origin(field.type) is Union
            and len(get_args(field.type)) == 2
            and get_args(field.type)[1] is type(None)
            and is_dataclass(get_args(field.type)[0])
        ):
            if data[field.name] is not None:
                data[field.name] = from_dict(get_args(field.type)[0], data[field.name])
    return datacls(**data)


def replace_factor_with_value(
    config: RawConfig,
    key: str,
    reference_value: int,
    bounds: Tuple[float, float],
) -> None:
    factor_key = key + '_factor'
    if factor_key not in config:
        assert key in config
    else:
        assert key not in config
        factor = config.pop(factor_key)
        assert bounds[0] <= factor <= bounds[1]
        config[key] = int(factor * reference_value)


def get_temporary_copy(path: Union[str, Path]) -> Path:
    path = env.get_path(path)
    assert not path.is_dir() and not path.is_symlink()
    tmp_path = path.with_name(
        path.stem + '___' + str(uuid.uuid4()).replace('-', '') + path.suffix
    )
    shutil.copyfile(path, tmp_path)
    atexit.register(lambda: tmp_path.unlink())
    return tmp_path


def get_python():
    python = Path('python3.9')
    return str(python) if python.exists() else 'python'

def get_catboost_config(real_data_path, is_cv=False):
    ds_name = Path(real_data_path).name
    C = load_json(f'tuned_models/catboost/{ds_name}_cv.json')
    return C

import os
import xml.etree.ElementTree as ET
import pandas as pd
import json
import csv

def parse_openml_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def strip_tag(tag):
        return tag.split('}', 1)[-1] if '}' in tag else tag

    data = {strip_tag(child.tag): child.text for child in root}
    return data

def read_csv_auto(filepath, encoding='utf-8', max_lines=5):

    delimiters = [',', ';', '\t', '|', ' ']
    
    with open(filepath, 'r', encoding=encoding) as f:
        sample = ''.join([next(f) for _ in range(max_lines)])

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=''.join(delimiters))
        delimiter = dialect.delimiter
    except Exception:
        for delimiter in delimiters:
            try:
                df = pd.read_csv(filepath, sep=delimiter, encoding=encoding)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
        raise ValueError("Could not determine the delimiter.")
    
    df = pd.read_csv(filepath, sep=delimiter, encoding=encoding)
    return df
import pandas as pd

def detect_and_clean_categoricals(
    df: pd.DataFrame, 
    max_unique_threshold: int = 20, 
    ratio_threshold: float = 0.05, 
    numeric_ratio: float = 0.9
):
    df_clean = df.copy()
    categorical_cols = []

    for col in df.columns:
        series = df[col]

        # 如果是 object，可能混合了数字和字符
        if series.dtype == 'object':
            series_numeric = pd.to_numeric(series, errors='coerce')
            valid_ratio = series_numeric.notna().mean()

            if valid_ratio > numeric_ratio:
                # 转成数值列，字符视为 NaN
                df_clean[col] = series_numeric
            else:
                # 保持分类变量
                categorical_cols.append(True)
                df_clean[col] = series.astype("category")
                continue

        # categorical 或 bool → 分类
        if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
            categorical_cols.append(True)
            df_clean[col] = series.astype("category")
            continue

        # 数值列 → 看 unique 数量
        if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
            n_unique = series.nunique(dropna=True)
            n_total = len(series.dropna())
            if n_total > 0 and (n_unique <= max_unique_threshold or (n_unique / n_total <= ratio_threshold)):
                categorical_cols.append(True)
                df_clean[col] = series.astype("category")
                continue
        
        categorical_cols.append(False)

    return df_clean, categorical_cols
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def preprocess_X(df, feature_cols):
    X_processed = pd.DataFrame(index=df.index)

    for col in feature_cols:
        series = df[col]
        if series.dtype == 'object' or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(series.astype(str).fillna("MISSING"))
        else:
            X_processed[col] = series.fillna(0)

    return X_processed

def rank_target_candidates(df: pd.DataFrame, categorical_cols: list, 
                           min_classes: int = 2, max_classes: int = 10, imbalance_threshold: float = 0.99):
    """
    返回所有可能的 target 候选，并按信息量评分排序
    """
    print("categorical_cols:", categorical_cols)
    candidates = []
    # feature_cols = [c for c in df.columns if c not in categorical_cols]  # 临时作为输入特征
    
    for col in categorical_cols:
        feature_cols = [c for c in df.columns if c != col]
        series = df[col].dropna()
        n_classes = series.nunique()
        # 类别数检查
        if n_classes < min_classes or n_classes > max_classes:
            continue
        
        # 类别分布检查
        value_counts = series.value_counts(normalize=True)
        if value_counts.iloc[0] > imbalance_threshold:
            continue
        
        # 计算互信息（粗略衡量这个列是否能被其他特征预测）
        try:
            X = preprocess_X(df, feature_cols)
            y = LabelEncoder().fit_transform(series)
            if X.shape[1] > 0:
                mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
                score = mi.mean()
            else:
                score = 0.0
        except Exception:
            score = 0.0
        
        candidates.append((col, n_classes, value_counts.iloc[0], score))
    
    # 按 (互信息高 → 类别数少 → 分布均衡) 排序
    candidates.sort(key=lambda x: (-x[3], x[1], x[2]))
    
    return candidates

def load_openml_dataset(id, cache_dir, load_categorical_indicator=False, detect_target_if_missing=True):
    '''
    Load an OpenML dataset from the cache.
    
    Parameters
    ----------
    - id: int, the ID of the dataset to load
    - cache_dir: str, the directory where the dataset is cached, e.g. "/Users/hanxinyan/Downloads/openml_cache/org/openml/www/datasets"
    - load_categorical_indicator: bool, whether to load the categorical indicator file or identify categorical columns automatically
    
    Returns
    ----------
    - df_filtered: pd.DataFrame, the dataset with ignored attributes removed
    - new_categorical_indicator: list of bool, indicating which columns are categorical
    - target_attribute: str, the name of the target attribute
    - description: str, the **raw** or cleaned dataset description
    '''
    dataset_dir = os.path.join(cache_dir, f"{id}")
    xml_path = os.path.join(dataset_dir, f"description.xml")
    csv_path = os.path.join(dataset_dir, f"dataset_{id}.csv")
    categorical_indicator_path = os.path.join(dataset_dir, f"categorical_indicator.json")
    
    
    pkl_df = read_csv_auto(csv_path)
    pkl_categorical_indicator = json.load(open(categorical_indicator_path, "r"))['categorical_indicator'] 
        
    pkl_df, auto_categorical_indicator = detect_and_clean_categoricals(pkl_df)
    print("auto_categorical_indicator:", auto_categorical_indicator)
    print("pkl_categorical_indicator:", pkl_categorical_indicator)
    if load_categorical_indicator is False:
        # 判断pkl_categorical_indicator是否都是False
        if all(x == False for x in pkl_categorical_indicator):
            categorical_indicator = auto_categorical_indicator
        else:
            categorical_indicator = pkl_categorical_indicator
    else:
        categorical_indicator = pkl_categorical_indicator
    info = parse_openml_xml(xml_path)
    target_attribute = info['default_target_attribute'] if 'default_target_attribute' in info else None

    if target_attribute is None and detect_target_if_missing:
        print("No default target attribute specified. Attempting to detect target candidates...")
        candidates = rank_target_candidates(pkl_df, [c for i, c in enumerate(pkl_df.columns) if categorical_indicator[i]])
        if candidates:
            target_attribute = candidates[0][0]
            print(f"Selected '{target_attribute}' as the target attribute.")
        else:
            print("No suitable target candidates found.")
    ignore_attributes = info['ignore_attribute'].split(',') if 'ignore_attribute' in info else []
    new_columns = []
    new_categorical_indicator = []
    for col, is_cat in zip(pkl_df.columns, categorical_indicator):
        if col not in ignore_attributes:
            new_columns.append(col)
            new_categorical_indicator.append(is_cat)
    df_filtered = pkl_df[new_columns]
    template_path = os.path.join(dataset_dir, "template.json")
    if os.path.exists(template_path):
        try:
            with open(template_path, "r") as f:
                template = json.load(f)
            return df_filtered, new_categorical_indicator, target_attribute, template
        except Exception as e:
            print(f"Failed to load template.json for dataset ID {id}: {e}")
            
            return df_filtered, new_categorical_indicator, target_attribute, info['description']
    else:
        desc = info['description']
        return df_filtered, new_categorical_indicator, target_attribute, desc

def load_kaggle_dataset(dataset_name, cache_dir):
    '''
    Load a kaggle dataset from the cache dir
    
    Parameters
    ----------
    - id: int, the ID of the dataset to load
    - cache_dir: str, the directory where the dataset is cached, e.g. "/Users/hanxinyan/Documents/kaggle/healthcare"

    Returns
    -------
    - df_filtered: pd.DataFrame, the dataset with ignored attributes removed
    - new_categorical_indicator: list of bool, indicating which columns are categorical
    - target_attribute: str, the name of the target attribute
    - description: str, the **cleaned** dataset description
    '''
    data_dir = os.path.join(cache_dir, dataset_name)
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the specified directory.")
    elif len(csv_files) > 1:
        print(f"Warning: Multiple CSV files found: {csv_files}. Using the first one: {csv_files[0]}")
    csv_file_path = os.path.join(data_dir, csv_files[0])
    df = read_csv_auto(csv_file_path)

    template_path = os.path.join(data_dir, "template.json")
    with open(template_path, "r") as f:
        template = json.load(f)

    target_attribute = list(template["target variable"].keys())[0]
    new_columns = []
    new_categorical_indicator = []
    for feature_dict in template["features"]:
        new_columns.append(list(feature_dict.keys())[0])
        if isinstance(list(feature_dict.values())[0], dict):
            new_categorical_indicator.append(True)
        else:
            new_categorical_indicator.append(False)
    new_columns.append(target_attribute)
    if isinstance(list(template["target variable"].values())[0], dict):
        new_categorical_indicator.append(True)
    else:
        new_categorical_indicator.append(False)
    
    df_filtered = df[new_columns]
    return df_filtered, new_categorical_indicator, target_attribute, template