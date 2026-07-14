# Feature Engineering pipeline
# Pre-processing script for the dataset
# Feature Engineering pipeline
# Pre-processing script for the dataset
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class FraudFeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.feature_cols = None

    def load_data(self, data_dir="data/raw"):
        log.info("Loading and merging data...")
        txn = pd.read_csv(f"{data_dir}/train_transaction.csv")
        idf = pd.read_csv(f"{data_dir}/train_identity.csv")
        df  = txn.merge(idf, on='TransactionID', how='left')
        log.info(f"Loaded {len(df):,} transactions, fraud rate: {df['isFraud'].mean():.3f}")
        return df

    def engineer_features(self, df):
        # TIME — fraud has strong night patterns
        df['hour']          = (df['TransactionDT'] / 3600).astype(int) % 24
        df['day_of_week']   = (df['TransactionDT'] / 86400).astype(int) % 7
        df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 4)).astype(int)
        df['is_weekend']    = (df['day_of_week'] >= 5).astype(int)

        # AMOUNT — fraudsters prefer specific patterns
        df['log_amount']     = np.log1p(df['TransactionAmt'])
        df['amount_rounded'] = (df['TransactionAmt'] % 1 == 0).astype(int)
        df['is_high_value']  = (df['TransactionAmt'] > 500).astype(int)

        # EMAIL — free email domains used more in fraud
        risky = ['gmail.com','yahoo.com','hotmail.com','outlook.com']
        df['high_risk_email'] = df['P_emaildomain'].isin(risky).astype(int)

        # CARD
        card_map = {'visa':0,'mastercard':1,'american express':2,'discover':3}
        df['card_type_enc'] = df['card4'].map(card_map).fillna(-1)
        return df

    def handle_missing(self, df):
        num = df.select_dtypes(include=[np.number]).columns
        cat = df.select_dtypes(include=['object']).columns
        df[num] = df[num].fillna(df[num].median())
        df[cat] = df[cat].fillna('unknown')
        return df

    def encode_cats(self, df, fit=True):
        for col in df.select_dtypes(include=['object']).columns:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[col] = df[col].map(
                        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
        return df

    def process(self, save=True):
        df = self.load_data()
        df = self.engineer_features(df)
        df = self.handle_missing(df)
        df = self.encode_cats(df, fit=True)
        excl = ['TransactionID','isFraud','TransactionDT']
        self.feature_cols = [c for c in df.columns if c not in excl]
        X, y = df[self.feature_cols], df['isFraud']
        if save:
            X.to_parquet('data/processed/X_train.parquet', index=False)
            y.to_parquet('data/processed/y_train.parquet', index=False)
        log.info(f"Done: {len(self.feature_cols)} features")
        return X, y

if __name__ == "__main__":
    FraudFeatureEngineer().process()