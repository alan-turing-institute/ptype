# [AIDA: Software package for ptype](https://github.com/alan-turing-institute/Hut23/issues/438)

Three FTE-month project to take ptype and make a more robust software package. See [project Kanban board](https://github.com/alan-turing-institute/ptype-dmkd/projects/1) for a more detailed view.

## Summary of high-level objectives

(✔ means done since last progress meeting)

| Objective | Description | Est. % | Completed | In progress/to do |
| --- | --- | --- | --- | --- |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype-dmkd/task:python-package)](https://github.com/alan-turing-institute/ptype-dmkd/labels/task:python-package) | <s>Python package with automated build & deployment</s> | 100% | [1](https://github.com/alan-turing-institute/ptype-dmkd/issues/1), [2](https://github.com/alan-turing-institute/ptype-dmkd/issues/2), [9](https://github.com/alan-turing-institute/ptype-dmkd/issues/9), [14](https://github.com/alan-turing-institute/ptype-dmkd/issues/14) | [10](https://github.com/alan-turing-institute/ptype-dmkd/issues/10), [31](https://github.com/alan-turing-institute/ptype-dmkd/issues/31) |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype-dmkd/task:test-coverage)](https://github.com/alan-turing-institute/ptype-dmkd/labels/task:test-coverage) | <s>Initial test coverage</s> | 100% | [3](https://github.com/alan-turing-institute/ptype-dmkd/issues/3), [4](https://github.com/alan-turing-institute/ptype-dmkd/issues/4), [6](https://github.com/alan-turing-institute/ptype-dmkd/issues/6), [7](https://github.com/alan-turing-institute/ptype-dmkd/issues/7), [38](https://github.com/alan-turing-institute/ptype-dmkd/issues/38), [54](https://github.com/alan-turing-institute/ptype-dmkd/issues/54), [60](https://github.com/alan-turing-institute/ptype-dmkd/issues/60) |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype-dmkd/task:core-api)](https://github.com/alan-turing-institute/ptype-dmkd/labels/task:core-api) | API to access, use and modify inferred schema | 70% | [11](https://github.com/alan-turing-institute/ptype-dmkd/issues/11), [37](https://github.com/alan-turing-institute/ptype-dmkd/issues/37), [44](https://github.com/alan-turing-institute/ptype-dmkd/issues/44), [62](https://github.com/alan-turing-institute/ptype-dmkd/issues/62), [92](https://github.com/alan-turing-institute/ptype-dmkd/issues/92)️, [96](https://github.com/alan-turing-institute/ptype-dmkd/issues/96), [118](https://github.com/alan-turing-institute/ptype-dmkd/issues/118)✔ | [117](https://github.com/alan-turing-institute/ptype-dmkd/issues/117), [123](https://github.com/alan-turing-institute/ptype-dmkd/issues/123), [135](https://github.com/alan-turing-institute/ptype-dmkd/issues/135) |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype-dmkd/task:internal-design)](https://github.com/alan-turing-institute/ptype-dmkd/labels/task:internal-design) | Apply engineering best practices | 80% | [93](https://github.com/alan-turing-institute/ptype-dmkd/issues/93), [116](https://github.com/alan-turing-institute/ptype-dmkd/issues/116) | [68](https://github.com/alan-turing-institute/ptype-dmkd/issues/68) |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype-dmkd/task:use-cases)](https://github.com/alan-turing-institute/ptype-dmkd/labels/task:use-cases) | Document API through example notebooks | 70% | [34](https://github.com/alan-turing-institute/ptype-dmkd/issues/34), [79](https://github.com/alan-turing-institute/ptype-dmkd/issues/79), [82](https://github.com/alan-turing-institute/ptype-dmkd/issues/82) | [78](https://github.com/alan-turing-institute/ptype-dmkd/issues/78), [83](https://github.com/alan-turing-institute/ptype-dmkd/issues/83), [86](https://github.com/alan-turing-institute/ptype-dmkd/issues/86), [88](https://github.com/alan-turing-institute/ptype-dmkd/issues/88) |

## Guiding principles

- Pandas dataframe as data format
- Provide analysis results as structured (meta)data
- Scriptable interface (avoid monolithic functionality)
- API driven by identified use cases
- Refactor/redesign as required to support easy implementation of other features
