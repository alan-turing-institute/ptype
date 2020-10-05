# Notebook testing

- We use `nbval` to test that notebook cell output remains unchanged. Notebook cell outputs need to be committed along with the notebook so that `nbval` can compare; this means all cells in the notebook need to be executed once in the appropriate order. 

- If you change the contents of a cell in such a way that the cell output also changes, you will need to reexecute the notebook before committing so that it contains the updated output. 

- `nbval` doesn't compare image data, so notebooks display output in more structured formats where possible.

- Cell contents should be computed deterministically, for reproducibility. For example, the behaviour of `set` is unlikely to remain consistent across system configurations or even runs of the same configuration.

- Evaluation of cells may also emit warnings to `stderr`. These are also unlikely to be stable across system configurations (e.g. because they contain absolute paths), so ideally cells should not produce warnings when they evaluate.

- `# NBVAL_IGNORE_OUTPUT` can be used to instruct `nbval` to ignore the output of a cell when checking for expected behaviour. This should be avoided unless absolutely necessary.
