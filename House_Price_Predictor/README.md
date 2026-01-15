# House Price Predictor (California Housing) 

This script:
- downloads/loads the housing dataset
- creates a stratified train/test split (based on income categories)
- preprocesses features (impute missing values + one-hot encode categories)
- trains a RandomForestRegressor
- reports cross-validation RMSE and test RMSE
- saves the trained pipeline to `housing_model.joblib`

## Project Structure
- `housing_train_minimal.py` — train + evaluate + save model
- `housing_model.joblib` — saved model (generated after training, not committed)

## Setup
```bash
pip install -r requirements.txt



