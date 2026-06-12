#!/usr/bin/env python3
import argparse, json, os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report
import joblib
import lightgbm as lgb

def load_data(path, sample_size=None):
    df = pd.read_csv(path)
    # try common title/label column names
    title_cols = [c for c in df.columns if 'title' in c.lower()]
    label_cols = [c for c in df.columns if 'category' in c.lower() or 'label' in c.lower()]
    if not title_cols or not label_cols:
        raise ValueError('Could not find title/label columns. Columns: {}'.format(df.columns.tolist()))
    df = df[[title_cols[0], label_cols[0]]].dropna()
    df.columns = ['title','label']
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)
    return df

def simple_text_pipeline():
    vect_w = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=20000)
    vect_c = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=5000)
    fe_union = FeatureUnion([('word', vect_w), ('char', vect_c)])
    return Pipeline([('fe', fe_union), ('clf', LogisticRegression(max_iter=200))])

def train(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = load_data(args.input, sample_size=args.sample_size)
    X = df['title'].astype(str)
    y = df['label'].astype(str)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if args.model == 'logreg':
        pipe = simple_text_pipeline()
        pipe.fit(X_train, y_train)
    elif args.model == 'lgb':
        # TF-IDF features -> LGB requires dense or sparse matrix; use word-only TFIDF for speed
        vect = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=20000)
        X_train_t = vect.fit_transform(X_train)
        X_val_t = vect.transform(X_val)
        lgbm = lgb.LGBMClassifier(n_estimators=200, random_state=42)
        lgbm.fit(X_train_t, y_train)
        pipe = Pipeline([('vect', vect), ('clf', lgbm)])
    else:
        raise ValueError('Unknown model: ' + args.model)

    preds = pipe.predict(X_val)
    report = classification_report(y_val, preds, output_dict=True)
    joblib.dump(pipe, args.out)
    meta = {'input': args.input, 'model': args.model, 'sample_size': int(args.sample_size or len(df)), 'report': report}
    with open(args.out + '.metadata.json','w') as f:
        json.dump(meta, f, indent=2)
    print('Saved model to', args.out)
    print('Validation classification report (macro avg):', report.get('macro avg', {}))

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--sample-size', type=int, default=5000)
    p.add_argument('--model', choices=['logreg','lgb'], default='logreg')
    args = p.parse_args()
    train(args)