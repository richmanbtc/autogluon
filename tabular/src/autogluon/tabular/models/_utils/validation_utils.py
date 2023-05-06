import os


def split_strict_val_fold(X, y, X_val, y_val, sample_weight, sample_weight_val):
    holdout_frac = float(os.getenv('AUTOGLUON_STRICT_VAL_HOLDOUT_FRAC', '0'))
    if holdout_frac == 0:
        with open('/tmp/my_autogluon_log', 'a') as f:
            print('strict val mode disabled', file=f)
        return X, y, X_val, y_val, sample_weight, sample_weight_val

    with open('/tmp/my_autogluon_log', 'a') as f:
        print('strict val mode used holdout_frac {}'.format(holdout_frac), file=f)
    fold_fit_size = int(X.shape[0] * (1.0 - holdout_frac))
    if sample_weight is not None:
        sample_weight, sample_weight_val = sample_weight[:fold_fit_size], sample_weight[fold_fit_size:]
    return (X[:fold_fit_size], y[:fold_fit_size],
            X[fold_fit_size:], y[fold_fit_size:],
            sample_weight, sample_weight_val)
