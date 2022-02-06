#!/bin/dash

## FIXME
input_path="/path/to/input/matrix/files/mtx/"

python3 run_CSR-row.py ${input_path}
python3 run_CSR-nnz.py ${input_path}
python3 run_COO-row.py ${input_path}
python3 run_COO-nnz-rgrn.py ${input_path}
python3 run_COO-nnz.py ${input_path}
python3 run_BCSR-block.py ${input_path}
python3 run_BCSR-nnz.py ${input_path}
python3 run_BCOO-block.py ${input_path}
python3 run_BCOO-nnz.py ${input_path}
