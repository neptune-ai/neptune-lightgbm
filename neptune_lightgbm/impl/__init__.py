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
]

import subprocess
import warnings
from io import BytesIO

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
from scikitplot.metrics import plot_confusion_matrix
from typing import Union

from neptune_lightgbm import __version__

try:
    # neptune-client=0.9.0+ package structure
    import neptune.new as neptune
    from neptune.new.internal.utils import verify_type
    from neptune.new.internal.utils.compatibility import expect_not_an_experiment
except ImportError:
    # neptune-client>=1.0.0 package structure
    import neptune
    from neptune.internal.utils import verify_type
    from neptune.internal.utils.compatibility import expect_not_an_experiment

INTEGRATION_VERSION_KEY = "source_code/integrations/neptune-lightgbm"


class NeptuneCallback:
    """Neptune callback for logging metadata during LightGBM model training.

    See guide with examples in the `Neptune-LightGBM docs`_.

    This callback logs parameters, evaluation results and info about the train_set:
    feature names, number of datapoints (``num_rows``) and number of features (``num_features``).

    Evaluation results are logged separately for every ``valid_sets``.
    For example with `"metric": "logloss"` and `valid_names=["train","valid"]`, 2 logs are created:
    `train/logloss` and `valid/logloss`.

    Callback works with ``lgbm.train()`` and ``lgbm.cv()`` functions, and with the sklearn API ``model.fit()``.
    For more details see `Neptune-LightGBM docs`_.

    Note:
        You can use public ``api_token="ANONYMOUS"`` and set ``project="common/lightgbm-integration"``
        for testing without registration.

    Args:
        run (:obj:`neptune.new.run.Run`): Neptune run object.
            A run in Neptune is a representation of all metadata that you log to Neptune.
            Learn more in `run docs`_.
        base_namespace(:obj:`str`, optional): Root namespace. All metadata will be logged inside.
            Default is empty string. In this case metadata is logged without common "base_namespace".

    Examples:
        For more examples visit `example scripts`_.

        Full script that does model training and logging of the metadata::

            import lightgbm as lgb
            import neptune.new as neptune
            from neptune.new.integrations.lightgbm import NeptuneCallback
            from sklearn.datasets import load_digits
            from sklearn.model_selection import train_test_split

            # Create run
            run = neptune.init(
                project="common/lightgbm-integration",
                api_token="ANONYMOUS",
                name="train-cls",
                tags=["lgbm-integration", "train", "cls"]
            )

            # Create neptune callback
            neptune_callback = NeptuneCallback(run=run)

            # Prepare data
            X, y = load_digits(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

            # Define parameters
            params = {
                "boosting_type": "gbdt",
                "objective": "multiclass",
                "num_class": 10,
                "metric": ["multi_logloss", "multi_error"],
                "num_leaves": 21,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "max_depth": 12,
            }

            # Train the model and log metadata to the run in Neptune
            gbm = lgb.train(
                params,
                lgb_train,
                num_boost_round=200,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=["training", "validation"],
                callbacks=[neptune_callback],
            )

    .. _Neptune-LightGBM docs:
        https://docs.neptune.ai/integrations-and-supported-tools/model-training/lightgbm
       _run docs:
        https://docs.neptune.ai/api-reference/run
       _example scripts:
        https://github.com/neptune-ai/examples/tree/main/integrations-and-supported-tools/lightgbm/scripts

    """

    def __init__(self, run: 'neptune.Run', base_namespace=""):
        expect_not_an_experiment(run)
        verify_type("run", run, neptune.Run)
        verify_type("base_namespace", base_namespace, str)

        if base_namespace and not base_namespace.endswith("/"):
            base_namespace += "/"

        self._run = run
        self._base_namespace = base_namespace
        self.params_logged = False
        self.feature_names_logged = False

        self._run[INTEGRATION_VERSION_KEY] = __version__

    def __call__(self, env):
        if not self.params_logged:
            self._run["{}params".format(self._base_namespace)] = env.params
            self._run["{}params/env/begin_iteration".format(self._base_namespace)] = env.begin_iteration
            self._run["{}params/env/end_iteration".format(self._base_namespace)] = env.end_iteration
            self.params_logged = True

        if not self.feature_names_logged:
            # lgb.train
            if isinstance(env.model, lgb.engine.Booster):
                self._run["{}feature_names".format(self._base_namespace)] = env.model.feature_name()
                self._run["{}train_set/num_features".format(self._base_namespace)] = env.model.train_set.num_feature()
                self._run["{}train_set/num_rows".format(self._base_namespace)] = env.model.train_set.num_data()
            # lgb.cv
            if isinstance(env.model, lgb.engine.CVBooster):
                for i, booster in enumerate(env.model.boosters):
                    self._run["{}/booster_{}/feature_names".format(self._base_namespace, i)] \
                        = booster.feature_name()
                    self._run["{}/booster_{}/train_set/num_features".format(self._base_namespace, i)] \
                        = booster.train_set.num_feature()
                    self._run["{}/booster_{}/train_set/num_rows".format(self._base_namespace, i)] \
                        = booster.train_set.num_feature()
            self.feature_names_logged = True

        # log metrics
        for row in env.evaluation_result_list:
            # lgb.train
            if len(row) == 4:
                dataset, metric, value, _ = row
                log_name = "{}{}/{}".format(self._base_namespace, dataset, metric)
                self._run[log_name].log(value, step=env.iteration)
            # lgb.cv
            if len(row) == 5:
                dataset, metric, value, _, std = row
                log_val_name = "{}{}/{}/val".format(self._base_namespace, dataset, metric)
                self._run[log_val_name].log(value, step=env.iteration)
                log_std_name = "{}{}/{}/std".format(self._base_namespace, dataset, metric)
                self._run[log_std_name].log(std, step=env.iteration)


def create_booster_summary(
    booster: Union[lgb.Booster, lgb.sklearn.LGBMModel],
    log_importances: bool = True,
    max_num_features: int = 10,
    list_trees: list = None,
    log_trees_as_dataframe: bool = True,
    log_pickled_booster: bool = True,
    log_trees: bool = False,
    tree_figsize: int = 30,
    log_confusion_matrix: bool = False,
    y_true: np.ndarray = None,
    y_pred: np.ndarray = None,
):
    """Create model summary after training that can be assigned to the run namespace.

    See guide with examples in the `Neptune-LightGBM docs`_.

    You can log multiple types of metadata:
        - pickled model
        - feature importance chart
        - visualized trees
        - trees represented as DataFrame
        - confusion matrix (only for classification problems)

    See Args section for more info how to parametrize behaviour of this function.

    Note:
        You can log summary to the new run, or to the same run that you used for logging model training.
        Second option can be very useful because you have all the information in the single run.

    Args:
        booster (:obj:`lgb.Booster` or :obj:`lgb.sklearn.LGBMModel`): Trained LightGBM model.
        log_importances (bool): Defaults to True. Log feature importance charts.
        max_num_features (int): Defaults to 10. Max number of top features on the importance charts.
            Works only if ``log_importances`` is set to ``True``.
            If None or <1, all features will be displayed.
            See `lightgbm.plot_importance`_ for details.
        list_trees (list): Defaults to None. Indices of the target tree to visualize.
            Works only if ``log_trees`` is set to ``True``.
            See `lightgbm.plot_tree`_ for details.
        log_trees_as_dataframe (bool): Defaults to True.
            Parse the model and log trees in the easy-to-read pandas DataFrame format.
            Works only for ``lgb.Booster``.
            See `lightgbm.Booster.trees_to_dataframe`_ for details.
        log_pickled_booster (bool): Defaults to True. Log model as pickled file.
        log_trees (bool): Defaults to False. Log visualized trees.
            This requires graphviz to work. Learn about setup in the `Neptune-LightGBM installation`_ docs.
        tree_figsize (int): Defaults to 30, Control size of the visualized tree image.
            Increase this in case you work with large trees.
            Works only if ``log_trees`` is set to ``True``.
        log_confusion_matrix (bool): Defaults to False. Log confusion matrix.
            If set to True, you need to pass ``y_true`` and ``y_pred``.
        y_true (:obj:`np.ndarray`): Defaults to None. True labels on the test set.
            Needed only if ``log_confusion_matrix`` is set to True.
        y_pred (:obj:`np.ndarray`): Defaults to None. Predictions on the test set.
            Needed only if ``log_confusion_matrix`` is set to True.

    Returns:
        dict: Python dictionary with all metadata, that can be assigned to the run namespace.
            ``run["booster_summary"] = create_booster_summary(...)``

    Examples:
        For more examples visit `example scripts`_.

        Full script that does logging during model training and logs booster summary after training::

            import lightgbm as lgb
            import neptune.new as neptune
            import numpy as np
            from neptune.new.integrations.lightgbm import NeptuneCallback, create_booster_summary
            from sklearn.datasets import load_digits
            from sklearn.model_selection import train_test_split

            # Create run
            run = neptune.init(
                project="common/lightgbm-integration",
                api_token="ANONYMOUS",
                name="train-cls",
                tags=["lgbm-integration", "train", "cls"]
            )

            # Create neptune callback
            neptune_callback = NeptuneCallback(run=run)

            # Prepare data
            X, y = load_digits(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

            # Define parameters
            params = {
                "boosting_type": "gbdt",
                "objective": "multiclass",
                "num_class": 10,
                "metric": ["multi_logloss", "multi_error"],
                "num_leaves": 21,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "max_depth": 12,
            }

            # Train the model and log metadata to the run in Neptune
            gbm = lgb.train(
                params,
                lgb_train,
                num_boost_round=200,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=["training", "validation"],
                callbacks=[neptune_callback],
            )

            y_pred = np.argmax(gbm.predict(X_test), axis=1)

            # Log summary metadata to the same run under the "lgbm_summary" namespace
            run["lgbm_summary"] = create_booster_summary(
                booster=gbm,
                log_trees=True,
                list_trees=[0, 1, 2, 3, 4],
                log_confusion_matrix=True,
                y_pred=y_pred,
                y_true=y_test
            )

    .. _Neptune-LightGBM docs:
        https://docs.neptune.ai/integrations-and-supported-tools/model-training/lightgbm
       _lightgbm.plot_importance:
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_importance.html#lightgbm-plot-importance
       _lightgbm.plot_tree:
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_tree.html#lightgbm-plot-tree
       _lightgbm.Booster.trees_to_dataframe:
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster.trees_to_dataframe
       _Neptune-LightGBM installation:
        https://docs.neptune.ai/integrations-and-supported-tools/model-training/lightgbm#install-requirements
       _example scripts:
        https://github.com/neptune-ai/examples/tree/main/integrations-and-supported-tools/lightgbm/scripts

    """
    results_dict = {}
    visuals_path = "visualizations/"
    if log_importances:
        split_plot = lgb.plot_importance(
            booster,
            importance_type="split",
            title="Feature importance (split)",
            max_num_features=max_num_features
        )
        gain_plot = lgb.plot_importance(
            booster,
            importance_type="gain",
            title="Feature importance (gain)",
            max_num_features=max_num_features
        )
        results_dict["{}feature_importances/split".format(visuals_path)] \
            = neptune.types.File.as_image(split_plot.figure)
        results_dict["{}feature_importances/gain".format(visuals_path)] \
            = neptune.types.File.as_image(gain_plot.figure)

    if log_trees:
        try:
            subprocess.call(["dot", "-V"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            log_trees = False
            message = "Graphviz executables not found, so trees will not be logged. " \
                      "Make sure the Graphviz executables are on your systems' PATH"
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
            trees_series.append(neptune.types.File.as_image(ax.figure))
        results_dict["{}trees".format(visuals_path)] = neptune.types.FileSeries(trees_series)

    if log_trees_as_dataframe:
        if isinstance(booster, lgb.Booster):
            df = booster.trees_to_dataframe()
            html_df = neptune.types.File.as_html(df)
            results_dict["trees_as_dataframe"] = html_df
            if not df.empty and not html_df.content:
                warnings.warn(
                    "'trees_as_dataframe' wasn't logged. Probably generated dataframe was to large.")
        else:
            warnings.warn("'trees_as_dataframe' won't be logged."
                          " `booster` must be instance of `lightgbm.Booster` class.")

    if log_pickled_booster:
        results_dict["pickled_model"] = neptune.types.File.as_pickle(booster)

    if log_confusion_matrix:
        ax = plot_confusion_matrix(y_true=y_true, y_pred=y_pred)
        results_dict[f"{visuals_path}confusion_matrix"] = neptune.types.File.as_image(ax.figure)

    return results_dict
