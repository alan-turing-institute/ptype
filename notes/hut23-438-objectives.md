# [AIDA: Software package for ptype](https://github.com/alan-turing-institute/Hut23/issues/438)

Three FTE-month project to take ptype and make a more robust software package. See [project Kanban board](https://github.com/alan-turing-institute/ptype/projects/1) for a more detailed view.

## Summary of high-level objectives

✔ means done since last progress meeting.

| Objective | Description | Est. % | Completed | In progress/to do | Deferred | Dropped |
| --- | --- | --- | --- | --- | --- | --- |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype/task:python-package)](https://github.com/alan-turing-institute/ptype/labels/task:python-package) | Python package with automated build & deployment | 100% | [1](https://github.com/alan-turing-institute/ptype/issues/1), [2](https://github.com/alan-turing-institute/ptype/issues/2), [9](https://github.com/alan-turing-institute/ptype/issues/9), [14](https://github.com/alan-turing-institute/ptype/issues/14), [10](https://github.com/alan-turing-institute/ptype/issues/10)✔, [140](https://github.com/alan-turing-institute/ptype/issues/140)✔ |  | [31](https://github.com/alan-turing-institute/ptype/issues/31) |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype/task:test-coverage)](https://github.com/alan-turing-institute/ptype/labels/task:test-coverage) | Initial test coverage | 90% | [3](https://github.com/alan-turing-institute/ptype/issues/3), [4](https://github.com/alan-turing-institute/ptype/issues/4), [6](https://github.com/alan-turing-institute/ptype/issues/6), [7](https://github.com/alan-turing-institute/ptype/issues/7), [38](https://github.com/alan-turing-institute/ptype/issues/38), [54](https://github.com/alan-turing-institute/ptype/issues/54), [60](https://github.com/alan-turing-institute/ptype/issues/60) | [127](https://github.com/alan-turing-institute/ptype/issues/127) |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype/task:core-api)](https://github.com/alan-turing-institute/ptype/labels/task:core-api) | API to access, use and modify inferred schema | 100% | [11](https://github.com/alan-turing-institute/ptype/issues/11), [37](https://github.com/alan-turing-institute/ptype/issues/37), [44](https://github.com/alan-turing-institute/ptype/issues/44), [62](https://github.com/alan-turing-institute/ptype/issues/62), [92](https://github.com/alan-turing-institute/ptype/issues/92)️, [96](https://github.com/alan-turing-institute/ptype/issues/96), [118](https://github.com/alan-turing-institute/ptype/issues/118)✔, [123](https://github.com/alan-turing-institute/ptype/issues/123), [136](https://github.com/alan-turing-institute/ptype/issues/136) |  | [117](https://github.com/alan-turing-institute/ptype/issues/117), [135](https://github.com/alan-turing-institute/ptype/issues/135)
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype/task:internal-design)](https://github.com/alan-turing-institute/ptype/labels/task:internal-design) | Apply engineering best practices | 100% | [93](https://github.com/alan-turing-institute/ptype/issues/93), [116](https://github.com/alan-turing-institute/ptype/issues/116), [68](https://github.com/alan-turing-institute/ptype/issues/68)✔ |  |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype/task:use-cases)](https://github.com/alan-turing-institute/ptype/labels/task:use-cases) | Document API through example notebooks | 90% | [34](https://github.com/alan-turing-institute/ptype/issues/34), [79](https://github.com/alan-turing-institute/ptype/issues/79), [82](https://github.com/alan-turing-institute/ptype/issues/82), [83](https://github.com/alan-turing-institute/ptype/issues/83) | [88](https://github.com/alan-turing-institute/ptype/issues/88) | [78](https://github.com/alan-turing-institute/ptype/issues/78) | [86](https://github.com/alan-turing-institute/ptype/issues/86) |

## Guiding principles

- Pandas dataframe as data format
- Provide analysis results as structured (meta)data
- Scriptable interface (avoid monolithic functionality)
- API driven by identified use cases
- Refactor/redesign as required to support easy implementation of other features
