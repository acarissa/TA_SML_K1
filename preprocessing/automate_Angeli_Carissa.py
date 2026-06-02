import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from joblib import dump

warnings.simplefilter(action='ignore', category=FutureWarning)

def preprocess_data(df, price_col, target_col, pipeline_save_path, column_header_path, train_save_path, test_save_path):
    """
    Full preprocessing pipeline
    """

    df = df.copy()

    # Step 1 — Create Price_Category from Price using quantile bins
    df[target_col] = pd.qcut(df[price_col], q=4, labels=False)
    print(f"\n── {target_col} distribution ──")
    print(df[target_col].value_counts().sort_index())

    # Step 2 — Drop original Price column (target is now Price_Category)
    df = df.drop(columns=[price_col])

    # Identify numeric and categorical feature columns (excluding target)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    # Save column headers (excluding target)
    feature_cols = [c for c in df.columns if c != target_col]
    pd.DataFrame(columns=feature_cols).to_csv(column_header_path, index=False)
    print(f"\nColumn headers saved : {column_header_path}")

    # Step 3 — Split raw data BEFORE any fitting (stratified to keep class balance)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Step 4 — Build pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # just in case for inference
        ('scaler',  StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # just in case for inference
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop'
    )

    # Step 5 — fit on train only, then transform each split separately
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed  = preprocessor.transform(X_test)

    # Recover column names after OHE expansion
    ohe_cols = []
    if cat_cols:
        ohe_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols).tolist()
    all_feature_cols = num_cols + ohe_cols

    # Step 6 — Save train.csv and test.csv (DataFrame only for saving)
    train_df = pd.DataFrame(X_train_transformed, columns=all_feature_cols)
    train_df[target_col] = y_train.values
    train_df.to_csv(train_save_path, index=False)
    print(f"Train set saved      : {train_save_path}  {train_df.shape}")

    test_df = pd.DataFrame(X_test_transformed, columns=all_feature_cols)
    test_df[target_col] = y_test.values
    test_df.to_csv(test_save_path, index=False)
    print(f"Test  set saved      : {test_save_path}  {test_df.shape}")

    # Step 7 — Save fitted pipeline (fitted on X_train only)
    dump(preprocessor, pipeline_save_path)
    print(f"Pipeline saved       : {pipeline_save_path}")

    # Return numpy arrays — ready to plug straight into any model
    return X_train_transformed, X_test_transformed, y_train, y_test


# MAIN
if __name__ == '__main__':
    DATA_PATH     = 'laptop_price_raw/Laptop_price.csv'
    PIPELINE_PATH = 'preprocessing/laptop_price_preprocessing/preprocessor_pipeline.joblib'
    HEADER_PATH   = 'preprocessing/laptop_price_preprocessing/column_headers.csv'
    TRAIN_PATH    = 'preprocessing/laptop_price_preprocessing/train.csv'
    TEST_PATH     = 'preprocessing/laptop_price_preprocessing/test.csv'

    os.makedirs('laptop_price_raw', exist_ok=True)
    os.makedirs('preprocessing/laptop_price_preprocessing', exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        price_col          = 'Price',
        target_col         = 'Price_Category',
        pipeline_save_path = PIPELINE_PATH,
        column_header_path = HEADER_PATH,
        train_save_path    = TRAIN_PATH,
        test_save_path     = TEST_PATH,
    )

    print("\n══ Split result ══")
    print(f"X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")
    print("\nX_train sample:")
    print(X_train[:3])
    print("\ny_train (Price_Category) sample:")
    print(y_train.head(3))