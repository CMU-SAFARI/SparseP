#!/bin/dash

## FIXME
input_path="/path/to/input/matrix/files/mtx/"

python3 run_DCSR.py  ${input_path}
python3 run_DCOO.py  ${input_path}
python3 run_DBCSR.py  ${input_path}
python3 run_DBCOO.py  ${input_path}

python3 run_RBDCSR.py  ${input_path}
python3 run_RBDCOO.py  ${input_path}
python3 run_RBDBCSR-block.py  ${input_path}
python3 run_RBDBCSR-nnz.py  ${input_path}
python3 run_RBDBCOO-block.py  ${input_path}
python3 run_RBDBCOO-nnz.py  ${input_path}

python3 run_BDCSR.py  ${input_path}
python3 run_BDCOO.py  ${input_path}
python3 run_BDBCSR-block.py  ${input_path}
python3 run_BDBCSR-nnz.py  ${input_path}
python3 run_BDBCOO-block.py  ${input_path}
python3 run_BDBCOO-nnz.py  ${input_path}

