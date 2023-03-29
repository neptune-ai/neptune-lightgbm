import lightgbm as lgb
import numpy as np
import pytest

try:
    from neptune import init_run
    from neptune.integrations.lightgbm import (
        NeptuneCallback,
        create_booster_summary,
    )
except ImportError:
    from neptune.new import init_run
    from neptune.new.integrations.lightgbm import (
        NeptuneCallback,
        create_booster_summary,
    )


args = (
    (True, True, True, True, True),
    (False, False, False, False, False),
)


@pytest.mark.parametrize("args", args)
def test_e2e(dataset, args):
    # Since these all arguments are independent of each other,
    # we don't check all combinations.
    log_importances, log_confusion_matrix, log_pickled_booster, log_trees, log_trees_as_dataframe = args
    # Start a run
    run = init_run()

    # Create a NeptuneCallback instance
    neptune_callback = NeptuneCallback(run=run)

    X_train, X_test, y_train, y_test = dataset

    lgb_train = lgb.Dataset(X_train, y_train)

    # Define model parameters
    params = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_class": 3,
    }

    # Train the model
    gbm = lgb.train(
        params,
        lgb_train,
        callbacks=[neptune_callback],
    )

    y_pred = np.argmax(gbm.predict(X_test), axis=1)

    run["summary"] = create_booster_summary(
        booster=gbm,
        log_trees=log_trees,
        log_trees_as_dataframe=log_trees_as_dataframe,
        list_trees=[0, 1],
        log_confusion_matrix=log_confusion_matrix,
        log_importances=log_importances,
        log_pickled_booster=log_pickled_booster,
        y_pred=y_pred,
        y_true=y_test,
    )

    run.wait()
    validate_results(
        run,
        log_importances,
        log_confusion_matrix,
        log_pickled_booster,
        log_trees,
        log_trees_as_dataframe,
        base_namespace="",
    )


def test_e2e_using_namespace(dataset):
    # NOTE: We don't create summary as summary
    # doesn't depend on base_namesace.

    # Start a run
    run = init_run()

    # Create a NeptuneCallback instance
    neptune_callback = NeptuneCallback(run=run, base_namespace="training")

    X_train, X_test, y_train, y_test = dataset

    lgb_train = lgb.Dataset(X_train, y_train)

    # Define model parameters
    params = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_class": 3,
    }

    # Train the model
    lgb.train(
        params,
        lgb_train,
        callbacks=[neptune_callback],
    )

    run.wait()
    validate_results(run, False, False, False, False, False, base_namespace="training")


def test_e2e_using_handler(dataset):
    # NOTE: We don't create summary as summary
    # doesn't depend on base_namesace.

    # Start a run
    run = init_run()

    # Create a NeptuneCallback instance with namespace handler
    neptune_callback = NeptuneCallback(run=run["namespace"], base_namespace="training")

    X_train, X_test, y_train, y_test = dataset

    lgb_train = lgb.Dataset(X_train, y_train)

    # Define model parameters
    params = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_class": 3,
    }

    # Train the model
    lgb.train(
        params,
        lgb_train,
        callbacks=[neptune_callback],
    )

    run.wait()
    validate_results(run, False, False, False, False, False, base_namespace="namespace/training")


def validate_results(
    run, log_importances, log_confusion_matrix, log_pickled_booster, log_trees, log_trees_as_dataframe, base_namespace
):
    assert run.exists(f"{base_namespace}/feature_names")
    assert run.exists(f"{base_namespace}/params")
    assert run[f"{base_namespace}/params/boosting_type"].fetch() == "gbdt"
    assert run[f"{base_namespace}/params/num_class"].fetch() == 3
    assert run[f"{base_namespace}/params/objective"].fetch() == "multiclass"
    assert run.exists(f"{base_namespace}/train_set")

    if log_importances:
        assert run.exists("summary/visualizations/feature_importances/split")
        assert run.exists("summary/visualizations/feature_importances/gain")
    else:
        assert not run.exists("summary/visualizations/feature_importances/split")
        assert not run.exists("summary/visualizations/feature_importances/gain")

    if log_confusion_matrix:
        assert run.exists("summary/visualizations/confusion_matrix")
    else:
        assert not run.exists("summary/visualizations/confusion_matrix")

    if log_pickled_booster:
        assert run.exists("summary/pickled_model")
    else:
        assert not run.exists("summary/pickled_model")

    if log_trees:
        assert run.exists("summary/visualizations/trees")
    else:
        assert not run.exists("summary/visualizations/trees")

    if log_trees_as_dataframe:
        assert run.exists("summary/trees_as_dataframe")
    else:
        assert not run.exists("summary/trees_as_dataframe")
