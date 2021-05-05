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
from io import BytesIO
from typing import Union

import lightgbm as lgb
import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
from matplotlib import image
from neptune.new.internal.utils import verify_type
from scikitplot.metrics import plot_confusion_matrix

from neptune_lightgbm import __version__


class NeptuneCallback:
    """Callable class meant for logging lightGBM learning curves to Neptune.
    Goes over the list of metrics and valid_sets passed to the `lgb.train`
    object and logs them to a separate channels. For example with 'objective': 'multiclass'
    and `valid_names=['train','valid']` there will be 2 channels created:
    `train_multiclass_logloss` and `valid_multiclass_logloss`.
    Object of this class should be passed to the `callbacks` parameter of the `lgb.train` function.
    Args:
        run(`neptune.new.run.Run`): Neptune Run. If this parameter is skipped then the last created Neptune Run in this process will be used.
        base_namespace(str): Prefix that should be added before the `metric_name`
            and `valid_name` before logging to the appropriate channel.
    Examples:
        Prepare dataset::
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            from sklearn.datasets import load_wine
            data = load_wine()
            X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1)
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        Define model parameters::
            params = {'boosting_type': 'gbdt',
                      'objective': 'multiclass',
                      'num_class': 3,
                      'num_leaves': 31,
                      'learning_rate': 0.05,
                      'feature_fraction': 0.9
                      }
        Initialize Neptune Run::
            run = neptune.init()
        Run `lgb.train` passing `NeptuneCallback` to the `callbacks` parameter::
            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=500,
                            valid_sets=[lgb_train, lgb_eval],
                            valid_names=['train','valid'],
                            callbacks=[NeptuneCallback(run)],
                           )
    Note:
        If you are running a k-fold validation it is a good idea to add the k-fold prefix
        and pass it to the `NeptuneCallback` constructor::
            prefix='fold_{}'.format(fold_id)
            monitor = NeptuneCallback(base_namespace=prefix)
    """

    def __init__(self, run: 'neptune.Run', base_namespace=''):
        verify_type('run', run, neptune.Run)
        verify_type('base_namespace', base_namespace, str)

        if base_namespace and not base_namespace.endswith('/'):
            base_namespace += '/'

        self._run = run
        self._base_namespace = base_namespace

        self._run['source_code/integrations/neptune-lightgbm'] = __version__

    def __call__(self, env):
        # eval_train
        # eval_valid
        # TODO: ('cv_agg', k, np.mean(v), metric_type[k], np.std(v))
        for name, loss_name, loss_value, *_ in env.evaluation_result_list:
            channel_name = '{}{}_{}'.format(self._base_namespace, name, loss_name)
            self._run[channel_name].log(loss_value, step=env.iteration)


def create_booster_summary(
        booster: Union[lgb.Booster, lgb.sklearn.LGBMModel],
        log_importances: bool = True,
        max_num_features: int = 10,
        list_trees: list = None,
        log_trees_as_dataframe: bool = True,  # works only for lgb.Booster
        log_pickled_booster: bool = True,
        log_trees: bool = False,  # requires graphviz
        log_confusion_matrix: bool = False,  # requires scikit-plot
        y_true: np.ndarray = None,
        y_pred: np.ndarray = None,
):
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
        trees_series = []
        for i in list_trees:
            digraph = lgb.create_tree_digraph(booster, tree_index=i, show_info='data_percentage')
            _, ax = plt.subplots(1, 1)
            s = BytesIO()
            s.write(digraph.pipe(format='png'))
            s.seek(0)
            ax.imshow(image.imread(s))
            ax.axis('off')
            trees_series.append(neptune.types.File.as_image(ax.figure))
        results_dict["{}trees".format(visuals_path)] = neptune.types.FileSeries(trees_series)

    if log_trees_as_dataframe and isinstance(booster, lgb.Booster):
        df = booster.trees_to_dataframe()
        results_dict["trees_as_dataframe"] = neptune.types.File.as_html(df)

    if log_pickled_booster:
        results_dict["pickled_model"] = neptune.types.File.as_pickle(booster)

    if log_confusion_matrix:
        ax = plot_confusion_matrix(y_true=y_true, y_pred=y_pred)
        results_dict["{}confusion_matrix".format(visuals_path)] = neptune.types.File.as_image(ax.figure)

    return results_dict
