# Storing notebooks in Git

To avoid spurious diffs and unwanted builds, `.git/config` defines a filter called `strip-notebook-output` which runs `nbconvert` to strip the output embedded in any `.ipynb` files prior to commands like git `diff` and `git commit`. 

`jupyter` needs to be on the path. If the `nbconvert` step fails for this or any other reason, the file will be staged with the output included. See https://stackoverflow.com/questions/28908319.
