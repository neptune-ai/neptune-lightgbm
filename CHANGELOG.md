## neptune-lightgbm 2.0.0

### Changes
- Removed `neptune` and `neptune-client` from base requirements. ([#22](https://github.com/neptune-ai/neptune-lightgbm/pull/22))

## neptune-lightgbm 1.0.0

 ### Changes
 - `NeptuneCallback` now accepts a namespace `Handler` as an alternative to `Run` for the `run` argument. This means that
   you can call it like `NeptuneCallback(run=run["some/namespace/"])` to log everything to the `some/namespace/`
   location of the run.
 - Removed size limit for `log_trees_as_dataframe` ([#14](https://github.com/neptune-ai/neptune-lightgbm/pull/14))

 ### Breaking changes
 - Instead of the `log()` method, the integration now uses `append()` which is available since version 0.16.14
   of neptune-client.

## neptune-lightgbm 0.10.0

### Changes
- Moved `neptune-lightgbm` package to `src` directory ([#8](https://github.com/neptune-ai/neptune-lightgbm/pull/8))
- Defaulted `log_trees_as_dataframe` to `false` ([#5](https://github.com/neptune-ai/neptune-lightgbm/pull/5))
- Updated `create_booster_summary` to log tree dataframe as a csv ([#5](https://github.com/neptune-ai/neptune-lightgbm/pull/5))
- Poetry as a package builder ([#13](https://github.com/neptune-ai/neptune-lightgbm/pull/13))

## Fixes

- Fixed NeptuneCallback import error - now possible to directly import with `from neptune_lightgbm import NeptuneCallback`
  ([#10](https://github.com/neptune-ai/neptune-lightgbm/pull/10))

## neptune-lightgbm 0.9.14

### Changes

- Changed integrations utils to be imported from non-internal package ([#6](https://github.com/neptune-ai/neptune-lightgbm/pull/6))

## neptune-lightgbm 0.9.13

### Features

- Mechanism to prevent using legacy Experiments in new-API integrations ([#4](https://github.com/neptune-ai/neptune-lightgbm/pull/4))
