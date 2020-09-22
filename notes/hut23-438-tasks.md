# [AIDA: Software package for ptype](https://github.com/alan-turing-institute/Hut23/issues/438)

Three FTE-month project to take ptype and make a more robust software package. See [project Kanban board](https://github.com/alan-turing-institute/ptype-dmkd/projects/1) for a more detailed view.

## Summary of tasks

| Task | Description | Est. % | Completed | In progress/to do |
| --- | --- | --- | --- | --- |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype-dmkd/task:python-package)](https://github.com/alan-turing-institute/ptype-dmkd/labels/task:python-package) | <s>Python package with automated build & deployment</s> | 100% | [1](https://github.com/alan-turing-institute/ptype-dmkd/issues/1), [2](https://github.com/alan-turing-institute/ptype-dmkd/issues/2), [9](https://github.com/alan-turing-institute/ptype-dmkd/issues/9), [14](https://github.com/alan-turing-institute/ptype-dmkd/issues/14) |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype-dmkd/task:test-coverage)](https://github.com/alan-turing-institute/ptype-dmkd/labels/task:test-coverage) | <s>Initial test coverage</s> | 100% | [3](https://github.com/alan-turing-institute/ptype-dmkd/issues/3), [4](https://github.com/alan-turing-institute/ptype-dmkd/issues/4), [6](https://github.com/alan-turing-institute/ptype-dmkd/issues/6), [7](https://github.com/alan-turing-institute/ptype-dmkd/issues/7), [38](https://github.com/alan-turing-institute/ptype-dmkd/issues/38), [54](https://github.com/alan-turing-institute/ptype-dmkd/issues/54) |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype-dmkd/task:core-api)](https://github.com/alan-turing-institute/ptype-dmkd/labels/task:core-api) | Expose analysis as structured metadata | 50% | | [11](https://github.com/alan-turing-institute/ptype-dmkd/issues/11), [37](https://github.com/alan-turing-institute/ptype-dmkd/issues/37), [62](https://github.com/alan-turing-institute/ptype-dmkd/issues/62) |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype-dmkd/task:categorical-data)](https://github.com/alan-turing-institute/ptype-dmkd/labels/task:categorical-data) | Extend with categorical data type inference | 50% | [44](https://github.com/alan-turing-institute/ptype-dmkd/issues/44)
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype-dmkd/task:internal-design)](https://github.com/alan-turing-institute/ptype-dmkd/labels/task:internal-design) | Reengineer into idiomatic Python | 50% | | [68](https://github.com/alan-turing-institute/ptype-dmkd/issues/68) |
| [![](https://img.shields.io/github/labels/alan-turing-institute/ptype-dmkd/task:usage-docs)](https://github.com/alan-turing-institute/ptype-dmkd/labels/task:usage-docs) | Document API through example notebooks | 25% | [34](https://github.com/alan-turing-institute/ptype-dmkd/issues/34) | [10](https://github.com/alan-turing-institute/ptype-dmkd/issues/10) |

## Guiding principles

- Pandas dataframe data format
- Provide analysis results as structured metadata
- Allow for easy end-user scripting (avoid black box functionality)
- API driven by identified use cases
