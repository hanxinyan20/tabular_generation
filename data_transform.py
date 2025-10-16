
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
def infer_dtypes(df:pd.DataFrame, max_unique_values_for_categorical=20):
    categorical_cols = []
    numerical_cols = []
    y_col = df.columns[-1]
    for col in df.columns[:-1]:
        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
            nunique_values = df[col].nunique(dropna=True)
            if nunique_values <= max_unique_values_for_categorical:
                categorical_cols.append(col)
                df[col] = df[col].astype('category')
            else:
                numerical_cols.append(col)
                df[col] = df[col].astype('float32')
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            categorical_cols.append(col)
        else:
            raise ValueError(f"Unknown dtype: {df[col].dtype} for column {col}")

    return categorical_cols, numerical_cols, y_col

def fill_missing_values(df:pd.DataFrame, categorical_cols, numerical_cols, impute_numerical_strategy='None', impute_categorical_strategy='missing'):
    new_df = df.copy()
    for col in categorical_cols:
        if impute_categorical_strategy == 'missing':
            if pd.api.types.is_categorical_dtype(new_df[col]):
                if 'missing' not in new_df[col].cat.categories:
                    new_df[col] = new_df[col].cat.add_categories(['missing'])
                new_df[col] = new_df[col].fillna('missing')
            else:
                new_df[col] = new_df[col].fillna('missing')
            new_df[col] = new_df[col].astype('str')
        elif impute_categorical_strategy == 'mode':
            mode_value = new_df[col].mode()[0]
            new_df[col] = new_df[col].fillna(mode_value)
        else:
            raise ValueError(f"Unknown impute_categorical_strategy: {impute_categorical_strategy}")
    for col in numerical_cols:
        if impute_numerical_strategy == 'mean':
            mean_value = new_df[col].mean()
            new_df[col] = new_df[col].fillna(mean_value)
        elif impute_numerical_strategy == 'median':
            median_value = new_df[col].median()
            new_df[col] = new_df[col].fillna(median_value)
        elif impute_numerical_strategy == 'zero':
            new_df[col] = new_df[col].fillna(0)
        elif impute_numerical_strategy == 'None':
            pass
        else:
            raise ValueError(f"Unknown impute_numerical_strategy: {impute_numerical_strategy}")
    return new_df

class XPreprocessor:
    def __init__(self, categorical_cols=None, numerical_cols=None,
                 cat_strategy='one-hot', num_strategy='standard'):
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.cat_strategy = cat_strategy
        self.num_strategy = num_strategy
        self.num_numerical = 0
        self.num_categories = []
        self.encoders = {}  
        self.feature_order = []       # 记录fit时的列顺序（按X.columns遍历）

    def fit(self, X: pd.DataFrame):
        # 记录列顺序
        self.feature_order = list(X.columns)

        for col in X.columns:
            if col in self.categorical_cols:
                if self.cat_strategy == 'one-hot':
                    # sklearn >=1.2 用 sparse_output；兼容老版本用 try
                    try:
                        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    except TypeError:
                        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
                    ohe.fit(X[[col]])
                    cats = ohe.categories_[0]
                    new_cols = [f"{col}__{c}" for c in cats]
                    self.encoders[col] = {'type': 'ohe', 'encoder': ohe, 'cols': new_cols}
                    self.num_categories.append(len(cats))
                elif self.cat_strategy == 'label':
                    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    oe.fit(X[[col]])
                    self.encoders[col] = {'type': 'label', 'encoder': oe, 'cols': [col]}
                    self.num_categories.append(len(oe.categories_[0]))
                else:
                    raise ValueError(f"Unknown cat_strategy={self.cat_strategy}")

            elif col in self.numerical_cols:
                if self.num_strategy == 'standard':
                    scaler = StandardScaler()
                    scaler.fit(X[[col]])
                    self.encoders[col] = {'type': 'scaler', 'encoder': scaler, 'cols': [col]}
                    self.num_numerical += 1
                elif self.num_strategy == 'min-max':
                    scaler = MinMaxScaler()
                    scaler.fit(X[[col]])
                    self.encoders[col] = {'type': 'scaler', 'encoder': scaler, 'cols': [col]}
                    self.num_numerical += 1
                elif self.num_strategy == 'quantile_normal':
                    scaler = QuantileTransformer(output_distribution='normal')
                    scaler.fit(X[[col]])
                    self.encoders[col] = {'type': 'scaler', 'encoder': scaler, 'cols': [col]}
                    self.num_numerical += 1
                elif self.num_strategy == 'quantile_uniform':
                    scaler = QuantileTransformer(output_distribution='uniform')
                    scaler.fit(X[[col]])
                    self.encoders[col] = {'type': 'scaler', 'encoder': scaler, 'cols': [col]}
                    self.num_numerical += 1
                elif self.num_strategy == 'none':
                    self.encoders[col] = {'type': 'passthrough', 'cols': [col]}
                else:
                    raise ValueError(f"Unknown num_strategy={self.num_strategy}")

            else:
                raise ValueError(f"Column {col} not in categorical_cols or numerical_cols")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # 保证X包含fit时的所有列
        missing = [c for c in self.feature_order if c not in X.columns]
        if missing:
            raise KeyError(f"X is missing columns seen in fit: {missing}")

        out_parts = []
        out_cols = []

        for col in self.feature_order:
            info = self.encoders[col]
            if info['type'] == 'ohe':
                arr = info['encoder'].transform(X[[col]])
                out_parts.append(arr)
                out_cols.extend(info['cols'])
            elif info['type'] == 'label':
                arr = info['encoder'].transform(X[[col]])
                out_parts.append(arr)
                out_cols.extend(info['cols'])  # [col]
            elif info['type'] == 'scaler':
                arr = info['encoder'].transform(X[[col]])
                out_parts.append(arr)
                out_cols.extend(info['cols'])  # [col]
            elif info['type'] == 'passthrough':
                arr = X[[col]].to_numpy()
                out_parts.append(arr)
                out_cols.extend(info['cols'])  # [col]
            else:
                raise ValueError(f"Unknown encoder type for column {col}: {info['type']}")

        data = np.hstack(out_parts) if len(out_parts) > 1 else out_parts[0]
        # 返回 DataFrame，带列名和原索引
        return pd.DataFrame(data=data, columns=out_cols, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_trans: pd.DataFrame) -> pd.DataFrame:
  
        if not isinstance(X_trans, pd.DataFrame):
            raise TypeError("inverse_transform() only accepts pandas.DataFrame input.")

        expected_cols = []
        for col in self.feature_order:
            expected_cols.extend(self.encoders[col]['cols'])
        missing = [c for c in expected_cols if c not in X_trans.columns]
        if missing:
            raise KeyError(f"Missing transformed columns in input: {missing}")
        if set(X_trans.columns) != set(expected_cols):
            raise ValueError("Input columns do not match expected transformed columns.")
        cols_back = []
        for col in self.feature_order:
            info = self.encoders[col]
            cols_here = info['cols']
            X_slice = X_trans[cols_here]

            if info['type'] == 'ohe':
                inv = info['encoder'].inverse_transform(X_slice)
                inv = pd.Series(inv.ravel(), index=X_trans.index, name=col)

            elif info['type'] == 'label':
                inv = info['encoder'].inverse_transform(X_slice)
                inv = pd.Series(inv.ravel(), index=X_trans.index, name=col)

            elif info['type'] == 'scaler':
                inv = info['encoder'].inverse_transform(X_slice)
                inv = pd.Series(inv.ravel(), index=X_trans.index, name=col)

            elif info['type'] == 'passthrough':
                inv = X_slice.iloc[:, 0].copy()
                inv.name = col

            else:
                raise ValueError(f"Unknown encoder type for column {col}: {info['type']}")

            cols_back.append(inv)

        X_inv = pd.concat(cols_back, axis=1)
        X_inv = X_inv[self.feature_order]
        return X_inv


class YPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='label', max_classes=None, other_label="__OTHER__", random_state=None):
        self.strategy = strategy
        self.max_classes = max_classes
        self.other_label = other_label
        self.random_state = random_state
        self._fitted = False

    @staticmethod
    def _ensure_1d(y):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = np.asarray(y).ravel()
        else:
            y = np.asarray(y).ravel()
        return y

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("YPreprocessor is not fitted yet.")

    def fit(self, y):
        y = self._ensure_1d(y)
        vals, counts = np.unique(y, return_counts=True)

        if self.max_classes is None:
            # 无上限：编码空间大小=真实类别数
            self.classes_ = list(vals)
            self.num_classes_ = len(self.classes_)
            code_space = self.num_classes_
        else:
            M = int(self.max_classes)
            if len(vals) <= M:
                self.classes_ = list(vals)
                self.num_classes_ = len(self.classes_)  # 真实类别数
                code_space = M                           # 编码空间=Max
            else:
                # 合并：Top(M-1) + other_label
                order = np.argsort(-counts)
                top_idx = order[: M - 1]
                kept = vals[top_idx].tolist()
                self.classes_ = kept + [self.other_label]
                self.num_classes_ = M
                code_space = M

        self.class_to_base_ = {c: i for i, c in enumerate(self.classes_)}
        rng = np.random.RandomState(self.random_state)
        perm = np.arange(code_space)
        rng.shuffle(perm)

        C = len(self.classes_)
        self.base_to_code_ = np.full(code_space, fill_value=-1, dtype=int) 
        self.class_to_code_ = {}        
        self.code_to_class_ = {}        

        for base_idx, code_idx in zip(range(C), perm[:C]):
            cls = self.classes_[base_idx]
            self.class_to_code_[cls] = code_idx
            self.code_to_class_[code_idx] = cls
            self.base_to_code_[base_idx] = code_idx

        self.code_space_ = code_space  
        self.merged_ = (self.max_classes is not None and len(vals) > self.max_classes)
        self._fitted = True
        return self

    def transform(self, y):
        self._check_fitted()
        y = self._ensure_1d(y)

        idx = np.empty(len(y), dtype=int)
        for i, v in enumerate(y):
            if v in self.class_to_code_:
                idx[i] = self.class_to_code_[v]
            else:
                if self.merged_:
                    idx[i] = self.class_to_code_[self.other_label]
                else:
                    raise ValueError(f"Unknown class '{v}' but no merging applied (max_classes not triggered).")

        if self.strategy == 'label':
            return idx
        elif self.strategy == 'one-hot':
            n, K = len(idx), self.code_space_
            out = np.zeros((n, K), dtype=np.float32)
            out[np.arange(n), idx] = 1.0
            return out
        else:
            raise ValueError(f"Unknown y strategy: {self.strategy}")

    def inverse_transform(self, y_trans):
        self._check_fitted()

        if self.strategy == 'label':
            codes = np.asarray(y_trans, dtype=int).ravel()
        elif self.strategy == 'one-hot':
            y_arr = np.asarray(y_trans)
            if y_arr.ndim != 2 or y_arr.shape[1] != self.code_space_:
                raise ValueError(f"one-hot shape must be (n, {self.code_space_}), got {y_arr.shape}")
            codes = y_arr.argmax(axis=1)
        else:
            raise ValueError(f"Unknown y strategy: {self.strategy}")

        out = np.empty(codes.shape[0], dtype=object)
        for i, c in enumerate(codes):
            cls = self.code_to_class_.get(int(c), None)
            if cls is None:
                out[i] = self.other_label
            else:
                out[i] = cls
        return out
    
def remove_missing_y_rows(df:pd.DataFrame, y_col:str):
    return df[df[y_col].notna()].reset_index(drop=True)

def sort_df_cols(df:pd.DataFrame, categorical_cols, numerical_cols, y_col):
    ordered_cols = numerical_cols + categorical_cols + [y_col]
    return df[ordered_cols]

def transform_df(df:pd.DataFrame, max_unique_values_for_categorical=20, impute_numerical_strategy='None', impute_categorical_strategy='missing', categorical_encoding_strategy='one-hot', numerical_scaling_strategy='standard', y_encoding_strategy='label', y_max_classes=None):
    categorical_cols, numerical_cols, y_col = infer_dtypes(df, max_unique_values_for_categorical)
    new_df = remove_missing_y_rows(df, y_col)
    new_df = sort_df_cols(new_df, categorical_cols, numerical_cols, y_col)
    new_df = fill_missing_values(new_df, categorical_cols, numerical_cols, impute_numerical_strategy, impute_categorical_strategy)
    X_preprocessor = XPreprocessor(
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        cat_strategy=categorical_encoding_strategy,
        num_strategy=numerical_scaling_strategy
    )
    X = new_df.drop(columns=[y_col])
    X_trans = X_preprocessor.fit_transform(X)
    y = new_df[y_col]
    y_preprocessor = YPreprocessor(strategy=y_encoding_strategy, max_classes=y_max_classes)
    y_trans = y_preprocessor.fit_transform(y)
    print("y_trans.shape:", y_trans.shape)
    return X_trans, y_trans, X_preprocessor, y_preprocessor