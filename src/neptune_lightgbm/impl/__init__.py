#
# Copyright (c) 2021, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

__all__ = [
    "create_booster_summary",
    "NeptuneCallback",
    "__version__",
]

import subprocess
import warnings
from io import BytesIO
from typing import Union

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
from scikitplot.metrics import plot_confusion_matrix

from neptune_lightgbm.impl.version import __version__

try:
    from neptune import Run
    from neptune.handler import Handler
    from neptune.integrations.utils import (
        expect_not_an_experiment,
        verify_type,
    )
    from neptune.types import (
        File,
        FileSeries,
    )
    from neptune.utils import stringify_unsupported
except ImportError:
    from neptune.new import Run
    from neptune.new.handler import Handler
    from neptune.new.integrations.utils import (
        expect_not_an_experiment,
        verify_type,
    )
    from neptune.new.types import (
        File,
        FileSeries,
    )
    from neptune.new.utils import stringify_unsupported

INTEGRATION_VERSION_KEY = "source_code/integrations/neptune-lightgbm"


class NeptuneCallback:
    """Neptune callback for logging metadata during LightGBM model training.

    This callback logs parameters, evaluation results and info about the train_set:
    feature names, number of datapoints (num_rows) and number of features (num_features).

    Evaluation results are logged separately for every valid_sets. For example,
    with `"metric": "logloss"` and `valid_names=["train", "valid"]`, two logs are created:
    `train/logloss` and `valid/logloss`.

    The callback works with the `lgbm.train()` and `lgbm.cv()` functions, and with
    `model.fit()` from the scikit-learn API.

    Args:
        run: Neptune run object. You can also pass a namespace handler object;
            for example, run["test"], in which case all metadata is logged under
            the "test" namespace inside the run.
        base_namespace: Root namespace where all metadata logged by the callback is stored.
            If omitted, the metadata is logged without a common root namespace.

    Example:

        import neptune

        # Create a Neptune run
        run = neptune.init_run()

        # Instantiate the callback and pass it to training function
        from neptune.integrations.lightgbm import NeptuneCallback

        neptune_callback = NeptuneCallback(run=run)
        gbm = lgb.train(params, ..., callbacks=[neptune_callback])

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/lightgbm
        API reference: https://docs.neptune.ai/api/integrations/lightgbm
    """

    def __init__(self, run: Union[Run, Handler], base_namespace: str = ""):
        expect_not_an_experiment(run)
        verify_type("run", run, (Run, Handler))
        verify_type("base_namespace", base_namespace, str)

        if base_namespace and not base_namespace.endswith("/"):
            base_namespace += "/"

        self._run = run
        self._base_namespace = base_namespace
        self.params_logged = False
        self.feature_names_logged = False

        root_obj = run
        if isinstance(run, Handler):
            root_obj = run.get_root_object()

        root_obj[INTEGRATION_VERSION_KEY] = __version__

    def __call__(self, env):
        if not self.params_logged:
            self._run[f"{self._base_namespace}params"] = stringify_unsupported(env.params)
            self._run[f"{self._base_namespace}params/env/begin_iteration"] = env.begin_iteration
            self._run[f"{self._base_namespace}params/env/end_iteration"] = env.end_iteration
            self.params_logged = True

        if not self.feature_names_logged:
            # lgb.train
            if isinstance(env.model, lgb.engine.Booster):
                self._run[f"{self._base_namespace}feature_names"] = stringify_unsupported(env.model.feature_name())
                self._run[f"{self._base_namespace}train_set/num_features"] = env.model.train_set.num_feature()
                self._run[f"{self._base_namespace}train_set/num_rows"] = env.model.train_set.num_data()
            # lgb.cv
            if isinstance(env.model, lgb.engine.CVBooster):
                for i, booster in enumerate(env.model.boosters):
                    self._run[f"{self._base_namespace}/booster_{i}/feature_names"] = stringify_unsupported(
                        booster.feature_name()
                    )

                    self._run[
                        f"{self._base_namespace}/booster_{i}/train_set/num_features"
                    ] = booster.train_set.num_feature()

                    self._run[
                        f"{self._base_namespace}/booster_{i}/train_set/num_rows"
                    ] = booster.train_set.num_feature()

            self.feature_names_logged = True

        # log metrics
        for row in env.evaluation_result_list:
            # lgb.train
            if len(row) == 4:
                dataset, metric, value, _ = row
                log_name = f"{self._base_namespace}{dataset}/{metric}"
                self._run[log_name].append(value, step=env.iteration)
            # lgb.cv
            if len(row) == 5:
                dataset, metric, value, _, std = row
                log_val_name = f"{self._base_namespace}{dataset}/{metric}/val"
                self._run[log_val_name].append(value, step=env.iteration)
                log_std_name = f"{self._base_namespace}{dataset}/{metric}/std"
                self._run[log_std_name].append(std, step=env.iteration)


def create_booster_summary(
    booster: Union[lgb.Booster, lgb.sklearn.LGBMModel],
    log_importances: bool = True,
    max_num_features: int = 10,
    list_trees: list = None,
    log_trees_as_dataframe: bool = False,
    log_pickled_booster: bool = True,
    log_trees: bool = False,
    tree_figsize: int = 30,
    log_confusion_matrix: bool = False,
    y_true: np.ndarray = None,
    y_pred: np.ndarray = None,
) -> dict:
    """Creates model summary after training that can be assigned to the run namespace.

     You can log multiple types of metadata:
     - pickled model
     - feature importance chart
     - visualized trees
     - trees represented as DataFrame
     - confusion matrix (only for classification problems)

    You can log the summary either to a new run or to the same run that you used during model training.

     Args:
         booster: Trained LightGBM model.
         log_importances: Whether to log feature importance charts.
         max_num_features: Max number of top features on the importance charts.
             Works only if log_importances is set to True.
             If 'None' or <1, all features will be displayed.
         list_trees: Indices of the target tree to visualize.
             Works only if log_trees is set to True.
         log_trees_as_dataframe: Parse the model and log trees in pandas DataFrame format.
             Works only for lgb.Booster.
         log_pickled_booster: Whether to log model as pickled file.
         log_trees: Whether to log visualized trees.
             Requires the Graphviz library to be installed.
         tree_figsize: Control size of the visualized tree image.
             Increase the value in case you work with large trees.
             Works only if log_trees is set to True.
         log_confusion_matrix: Whether to log confusion matrix.
             If set to True, you need to pass y_true and y_pred.
         y_true: True labels on the test set.
             Needed only if log_confusion_matrix is set to True.
         y_pred: Predictions on the test set.
             Needed only if log_confusion_matrix is set to True.

     Returns:
         Python dictionary that contains all the metadata and can be assigned to the run:
             `run["booster_summary"] = create_booster_summary(...)`

     Example:
         import neptune
         from neptune.integrations.lightgbm import create_booster_summary

         run = neptune.init_run()
         gbm = lgb.train(params, ...)
         run["lgbm_summary"] = create_booster_summary(booster=gbm)

     For more, see the docs:
         Tutorial: https://docs.neptune.ai/integrations/lightgbm
         API reference: https://docs.neptune.ai/api/integrations/lightgbm
    """
    results_dict = {}
    visuals_path = "visualizations/"
    if log_importances:
        split_plot = lgb.plot_importance(
            booster, importance_type="split", title="Feature importance (split)", max_num_features=max_num_features
        )
        gain_plot = lgb.plot_importance(
            booster, importance_type="gain", title="Feature importance (gain)", max_num_features=max_num_features
        )
        results_dict[f"{visuals_path}feature_importances/split"] = File.as_image(split_plot.figure)

        results_dict[f"{visuals_path}feature_importances/gain"] = File.as_image(gain_plot.figure)

    if log_trees:
        try:
            subprocess.call(["dot", "-V"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            log_trees = False
            message = (
                "Graphviz executables not found, so trees will not be logged. "
                "Make sure the Graphviz executables are on your systems' PATH"
            )
            warnings.warn(message)

    if log_trees:
        trees_series = []
        for i in list_trees:
            digraph = lgb.create_tree_digraph(booster, tree_index=i, show_info="data_percentage")
            _, ax = plt.subplots(1, 1, figsize=(tree_figsize, tree_figsize))
            s = BytesIO()
            s.write(digraph.pipe(format="png"))
            s.seek(0)
            ax.imshow(image.imread(s))
            ax.axis("off")
            trees_series.append(File.as_image(ax.figure))
        results_dict[f"{visuals_path}trees"] = FileSeries(trees_series)

    if log_trees_as_dataframe:
        if isinstance(booster, lgb.Booster):
            df = booster.trees_to_dataframe()
            stream_buffer = BytesIO()
            df.to_csv(stream_buffer, index=False)
            results_dict["trees_as_dataframe"] = File.from_stream(stream_buffer, extension="csv")
        else:
            warnings.warn(
                "'trees_as_dataframe' won't be logged." " `booster` must be instance of `lightgbm.Booster` class."
            )

    if log_pickled_booster:
        results_dict["pickled_model"] = File.as_pickle(booster)

    if log_confusion_matrix:
        ax = plot_confusion_matrix(y_true=y_true, y_pred=y_pred)
        results_dict[f"{visuals_path}confusion_matrix"] = File.as_image(ax.figure)

    return results_dict
