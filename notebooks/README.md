# Notebook testing

- We use `nbval` to test that notebook cell output remains unchanged. Notebook cell outputs need to be committed along with the notebook so that `nbval` can compare; this means all cells in the notebook need to be executed once in the appropriate order. 

- If you change the contents of a cell in such a way that the cell output also changes, you will need to reexecute the notebook before committing so that it contains the updated output. 

- `nbval` doesn't compare image data, so notebooks display output in more structured formats where possible.
