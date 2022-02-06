import os 
import sys
import glob
import getpass

result_path = "results/"

def run(input_path):
    
    NR_TASKLETS = [16]
    NR_DPUS = [1024, 2048]
    DATATYPES = ["int8", "int16", "int32", "int64"]
    MATRICES = ["ldoor.mtx", "af_shell1.mtx", "roadNet-TX.mtx", "parabolic_fem.mtx", "poisson3Db.mtx", "delaunay_n19.mtx", "com-Youtube.mtx", "pkustk14.mtx"]

    pwd = os.getcwd()
    path = pwd.replace("scripts/run_1D", "") + "spmv/1D/BCSR-block"
    os.chdir(path)
    os.makedirs(result_path, exist_ok=True)
    for file in MATRICES:
        print(file)
        run_cmd = "bin/spmv_host " + " -f " + input_path + file
        for dt in DATATYPES:
            for r in NR_DPUS:
                for t in NR_TASKLETS:
                    # Balance Blocks across tasklets
                    os.system("make clean")
                    make_cmd = "make NR_DPUS=" + str(r) + " NR_TASKLETS=" + str(t) + " TYPE=" + dt.upper() + " BLNC_TSKLT_BLOCK=1 BLNC_TSKLT_NNZ=0"
                    os.system(make_cmd)
                    temp_file = str(file[:-4]) 
                    temp_file = temp_file.replace("_", "-") 
                    r_cmd = run_cmd + " >> " + result_path + temp_file 
                    r_cmd = r_cmd + "_" + dt + "_tl" + str(t) + "_dpus" + str(r) +  "_block.out" 
                    os.system(r_cmd)


                    # Balance NNZs across tasklets
                    os.system("make clean")
                    make_cmd = "make NR_DPUS=" + str(r) + " NR_TASKLETS=" + str(t) + " TYPE=" + dt.upper() + " BLNC_TSKLT_BLOCK=0 BLNC_TSKLT_NNZ=1"
                    os.system(make_cmd)
                    r_cmd = run_cmd + " >> " + result_path + temp_file
                    r_cmd = r_cmd + "_" + dt + "_tl" + str(t) + "_dpus" + str(r) +  "_nnz.out" 
                    os.system(r_cmd)       



def main():
    input_path = sys.argv[1]
    run(input_path)

if __name__ == "__main__":
    main()
