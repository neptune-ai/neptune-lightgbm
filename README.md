# Neptune + LightGBM Integration

Experiment tracking for LightGBM-trained models.

## What will you get with this integration?

* Log, organize, visualize, and compare ML experiments in a single place
* Monitor model training live
* Version and query production-ready models and associated metadata (e.g., datasets)
* Collaborate with the team and across the organization

## What will be logged to Neptune?

* training and validation metrics,
* parameters,
* feature names, num_features, and num_rows for the train set,
* hardware consumption (CPU, GPU, memory),
* stdout and stderr logs,
* training code and Git commit information,
* [other metadata](https://docs.neptune.ai/logging/what_you_can_log)

![image](https://docs.neptune.ai/img/app/integrations/lightgbm.png)

## Resources

* [Documentation](https://docs.neptune.ai/integrations/lightgbm)
* [Code example on GitHub](https://github.com/neptune-ai/examples/blob/main/integrations-and-supported-tools/lightgbm/scripts/Neptune_LightGBM_train_summary.py)
* [Example of a run logged in the Neptune app](https://app.neptune.ai/o/common/org/lightgbm-integration/e/LGBM-86/dashboard/train-cls-summary-6c07f9e0-36ca-4432-9530-7fd3457220b6)
* [Run example in Google Colab](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/lightgbm/notebooks/Neptune_LightGBM.ipynb)

## Example

```
# On the command line:
pip install neptune-lightgbm
```

```python
# In Python:
import lightgbm as lgb
import neptune
from neptune.integrations.lightgbm import NeptuneCallback

# Start a run
run = neptune.init_run(
    project="common/lightgbm-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
)

# Create a NeptuneCallback instance
neptune_callback = NeptuneCallback(run=run)

# Prepare datasets
...
lgb_train = lgb.Dataset(X_train, y_train)

# Define model parameters
params = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "num_class": 10,
    ...
}

# Train the model
gbm = lgb.train(
    params,
    lgb_train,
    callbacks=[neptune_callback],
)
```

## Support

If you got stuck or simply want to talk to us, here are your options:

* Check our [FAQ page](https://docs.neptune.ai/getting-started/getting-help#frequently-asked-questions)
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! When in the Neptune application click on the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP),
* You can just shoot us an email at support@neptune.ai
