#!/usr/bin/env python3
import argparse, joblib
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--title', required=True)
    args = p.parse_args()
    pipe = joblib.load(args.model)
    pred = pipe.predict([args.title])
    probs = None
    if hasattr(pipe, 'predict_proba'):
        probs = pipe.predict_proba([args.title])[0]
    print('Title:', args.title)
    print('Predicted:', pred[0])
    if probs is not None:
        topk = sorted(zip(pipe.classes_, probs), key=lambda x: -x[1])[:5]
        for c,pv in topk:
            print(f'  {c}: {pv:.3f}')
if __name__=='__main__':
    main()