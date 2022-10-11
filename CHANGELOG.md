## [UNRELEASED] neptune-lightgbm 0.9.15

### Changes

- Moved `neptune-lightgbm` package to `src` directory ([#8](https://github.com/neptune-ai/neptune-lightgbm/pull/8))
- Updated `create_booster_summary` to log tree dataframe as a csv ([#5](https://github.com/neptune-ai/neptune-lightgbm/pull/5))

## Fixes

- Fixed NeptuneCallback import error - now possible to directly import with `from neptune_lightgbm import NeptuneCallback`
  ([#10](https://github.com/neptune-ai/neptune-lightgbm/pull/10))

## neptune-lightgbm 0.9.14

### Changes

- Changed integrations utils to be imported from non-internal package ([#6](https://github.com/neptune-ai/neptune-lightgbm/pull/6))

## neptune-lightgbm 0.9.13

### Features

- Mechanism to prevent using legacy Experiments in new-API integrations ([#4](https://github.com/neptune-ai/neptune-lightgbm/pull/4))
