/** Candidate lists captured from the production Optuna default search-space provider. */
export const pruningSearchSpaces: Record<string, Record<string, unknown[]>> = {
  "random_forest_classifier": {
    "n_estimators": [
      50,
      100,
      200,
      500
    ],
    "max_depth": [
      null,
      5,
      10,
      20,
      30,
      50
    ],
    "min_samples_split": [
      2,
      5,
      10,
      20
    ],
    "min_samples_leaf": [
      1,
      2,
      4,
      8
    ],
    "criterion": [
      "gini",
      "entropy",
      "log_loss"
    ],
    "bootstrap": [
      true,
      false
    ]
  },
  "ridge_regression": {
    "alpha": [
      0.01,
      0.1,
      1.0,
      10.0,
      100.0
    ],
    "solver": [
      "auto",
      "svd",
      "cholesky",
      "lsqr",
      "sparse_cg",
      "sag",
      "saga"
    ],
    "fit_intercept": [
      true,
      false
    ],
    "max_iter": [
      1000,
      2000,
      5000
    ]
  },
  "sgd_classifier": {
    "loss": [
      "log_loss",
      "hinge",
      "modified_huber"
    ],
    "penalty": [
      "l1",
      "l2"
    ],
    "alpha": [
      1e-05,
      0.0001,
      0.001,
      0.01
    ],
    "max_iter": [
      500,
      1000,
      2000
    ]
  },
  "xgboost_classifier": {
    "n_estimators": [
      100,
      200,
      500,
      1000
    ],
    "max_depth": [
      3,
      5,
      7,
      9
    ],
    "learning_rate": [
      0.01,
      0.05,
      0.1,
      0.3
    ],
    "subsample": [
      0.6,
      0.8,
      1.0
    ],
    "colsample_bytree": [
      0.6,
      0.8,
      1.0
    ],
    "min_child_weight": [
      1,
      3,
      5,
      7
    ],
    "reg_alpha": [
      0.0,
      0.01,
      0.1,
      1.0
    ],
    "reg_lambda": [
      0.1,
      1.0,
      5.0,
      10.0
    ]
  },
  "lgbm_classifier": {
    "n_estimators": [
      100,
      200,
      500,
      1000
    ],
    "num_leaves": [
      15,
      31,
      63,
      127,
      255
    ],
    "learning_rate": [
      0.01,
      0.05,
      0.1,
      0.2
    ],
    "max_depth": [
      -1,
      5,
      10,
      20
    ],
    "min_child_samples": [
      5,
      10,
      20,
      50
    ],
    "subsample": [
      0.6,
      0.8,
      1.0
    ],
    "colsample_bytree": [
      0.6,
      0.8,
      1.0
    ],
    "reg_alpha": [
      0.0,
      0.01,
      0.1,
      1.0
    ],
    "reg_lambda": [
      0.0,
      0.01,
      0.1,
      1.0
    ],
    "boosting_type": [
      "gbdt",
      "dart",
      "goss"
    ]
  }
};
