# From https://www.blog.pythonlibrary.org/2018/10/16/testing-jupyter-notebooks/.

import nbformat
import os
from nbconvert.preprocessors import ExecutePreprocessor
import sys

def run_notebook(notebook_path):
    nb_name, _ = os.path.splitext(os.path.basename(notebook_path))
    dirname = os.path.dirname(notebook_path)

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=600, kernel_name='python3')
    proc.allow_errors = True

    proc.preprocess(nb, {'metadata': {'path': '/'}})
    output_path = os.path.join(dirname, '{}_all_output.ipynb'.format(nb_name))

    with open(output_path, mode='wt') as f:
        nbformat.write(nb, f)
    errors = []
    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    errors.append(output)
    return nb, errors

if __name__ == '__main__':
   print(sys.argv)
   print(sys.argv[1])
   nb, errors = run_notebook(sys.argv[1])
   print(errors)