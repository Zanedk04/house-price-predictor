# imports
from __future__ import annotations
from pathlib import Path
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib




def stratified_split_by_income(
    housing_full: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    housing = housing_full.copy() # makes a copy so we dont modify original

    # Creates an income category from "median_income" column
    # Group median income into 5 buckets so we can keep a similar percentage of 
    # each category in train/test split
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], # bucket edges
        labels=[1, 2, 3, 4, 5], # bucket labels
    )

    # Split into train/test while preserving the proportion of each income category.
    # This makes the test set more representative of the full dataset.
    train_set, test_set = train_test_split(
        housing,
        test_size=test_size,
        stratify=housing["income_cat"],
        random_state=random_state,
    )

    # Remove the helper column so it doesn't accidentally get used as a feature.
    for set_ in (train_set, test_set):
        set_.drop(columns=["income_cat"], inplace=True)

    # Return train and test dataframes
    return train_set, test_set


def build_preprocessing() -> Pipeline:
    """Preprocess numeric + categorical columns (no hand-written column lists needed)."""
    # Numeric pipeline:
    # Fills missing numeric values using the column median
    # scales numbers; RandomForest doesn't need scaling, so it's commented out
    num_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            # Scaling is not required for RandomForest, but kept here 
            # ("standardize", StandardScaler()),
        ]
    )
     # Categorical pipeline:
    # Fills missing categories with the most common category
    # - One-hot encode categories into 0/1 columns
    # - ignores unseen categories at prediction time instead of erroring
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    # Apply numeric pipeline to numeric columns and categorical pipeline to object/category columns.
    # make_column_selector automatically chooses columns based on dtype.
    preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=["object", "category"])),
        remainder="drop",
    )
    return preprocessing


def main() -> None:
    # reads housing dataset
    housing_full = pd.read_csv("/Users/zanekelley/House_Price_Predictor/housing.csv")

    # splits data set into train/test split, test size being 20%
    train_set, test_set = stratified_split_by_income(housing_full, test_size=0.2, random_state=42)

    # Separate inputs (X) from target (y)
    # X - features used for predictions
    # y - labels we want to predict
    X_train = train_set.drop(columns=["median_house_value"]) 
    y_train = train_set["median_house_value"].copy() 

    X_test = test_set.drop(columns=["median_house_value"])
    y_test = test_set["median_house_value"].copy()

    # Builds the preprocessing transformer (numeric + categorical handling)
    preprocessing = build_preprocessing()

    # defines model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )


    # Combine preprocessing + model into one pipeline
    #ensures the exact same preprocessing is applied during training and prediction.
    full_pipeline = make_pipeline(preprocessing, model)

    # Cross-validation on training set:
    # trains/evaluates multiple times on different splits of the training data
    # gives a more reliable estimate than a single train/val split
    mse_scores = -cross_val_score(
        full_pipeline,
        X_train,
        y_train,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    # Convert MSE to RMSE, RMSE is in the same units as the target: dollars
    rmse_scores = np.sqrt(mse_scores)

    print(f"CV RMSE: mean={rmse_scores.mean():.0f}, std={rmse_scores.std():.0f}")

    # Train on full training set, then evaluate once on the test set
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {test_rmse:.0f}")


    # Save the entire pipeline (prep + model)
    out_path = Path("housing_model.joblib")
    joblib.dump(full_pipeline, out_path)
    print(f"Saved model to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
