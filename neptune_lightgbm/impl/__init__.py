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
import logging
from io import BytesIO
from typing import Union

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from graphviz import ExecutableNotFound as GraphvizExecutableNotFound
from matplotlib import image
from scikitplot.metrics import plot_confusion_matrix

from neptune_lightgbm import __version__

try:
    # neptune-client=0.9.0 package structure
    import neptune.new as neptune
    from neptune.new.internal.utils import verify_type
except ImportError:
    # neptune-client=1.0.0 package structure
    import neptune
    from neptune.internal.utils import verify_type

__all__ = [
    'NeptuneCallback',
    'create_booster_summary',
]

log = logging.getLogger()


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
        # TODO:
        # (data_name, eval_name, val, is_higher_better): `lightgbm.basic.Booster.__inner_eval`
        # ('cv_agg', k, np.mean(v), metric_type[k], np.std(v)): `lightgbm.engine._agg_cv_result`
        for name, loss_name, loss_value, *_ in env.evaluation_result_list:
            channel_name = 'foo/{}/foo2/{}_{}'.format(self._base_namespace, name, loss_name)
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

    # TODO: Your file is larger than 15MB. Neptune supports logging files in-memory objects smaller than 15MB.
    #       Resize or increase compression of this object

    if log_trees:
        try:
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
        except GraphvizExecutableNotFound:
            # TODO: link to Neptune docs section where will be such info: https://graphviz.org/download/
            log.warning(f"Trees won't be logged. To make it work install graphviz library on your OS.")

    if log_trees_as_dataframe:
        if isinstance(booster, lgb.Booster):
            df = booster.trees_to_dataframe()
            results_dict["trees_as_dataframe"] = neptune.types.File.as_html(df)
        else:
            # TODO: log here and don't perform action, or detect it when calculation starts and raise an exception
            log.warning("Trees won't be logged as dataframe. `booster` must be instance of `lightgbm.Booster` class.")

    if log_pickled_booster:
        results_dict["pickled_model"] = neptune.types.File.as_pickle(booster)

    if log_confusion_matrix:
        ax = plot_confusion_matrix(y_true=y_true, y_pred=y_pred)
        results_dict[f"{visuals_path}confusion_matrix"] = neptune.types.File.as_image(ax.figure)

    return results_dict
