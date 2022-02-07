import os 
import sys
import glob
import getpass

result_path = "results_1DPU/"

def run(input_path):
    
    NR_TASKLETS = [1, 2, 4, 8, 12, 16, 20, 24]
    NR_DPUS = [1]
    DATATYPES = ["int8", "int32", "int16", "int64", "fp32", "fp64"]
    MATRICES = ["wing_nodal.mtx", "delaunay_n13.mtx", "pkustk08.mtx", "raefsky4.mtx"]

    pwd = os.getcwd()
    path = pwd.replace("scripts/run_1D", "") + "spmv/1D/COO-nnz"
    os.chdir(path)
    os.makedirs(result_path, exist_ok=True)
    for file in MATRICES:
        print(file)
        run_cmd = "bin/spmv_host " + " -f " + input_path + file
        for dt in DATATYPES:
            for r in NR_DPUS:
                for t in NR_TASKLETS:
                    # Coarse-grained
                    os.system("make clean")
                    make_cmd = "make NR_DPUS=" + str(r) + " NR_TASKLETS=" + str(t) + " TYPE=" + dt.upper() + " CGLOCK=1 FGLOCK=0 LOCKFREE=0"
                    os.system(make_cmd)
                    temp_file = str(file[:-4]) 
                    temp_file = temp_file.replace("_", "-") 
                    r_cmd = run_cmd + " >> " + result_path + temp_file 
                    r_cmd = r_cmd + "_" + dt + "_tl" + str(t) + "_dpus" + str(r) +  "_cg.out" 
                    os.system(r_cmd)


                    # Fine-grained
                    os.system("make clean")
                    make_cmd = "make NR_DPUS=" + str(r) + " NR_TASKLETS=" + str(t) + " TYPE=" + dt.upper() + " CGLOCK=0 FGLOCK=1 LOCKFREE=0"
                    os.system(make_cmd)
                    r_cmd = run_cmd + " >> " + result_path + temp_file
                    r_cmd = r_cmd + "_" + dt + "_tl" + str(t) + "_dpus" + str(r) +  "_fg.out" 
                    os.system(r_cmd)       


                    # Lock-free
                    os.system("make clean")
                    make_cmd = "make NR_DPUS=" + str(r) + " NR_TASKLETS=" + str(t) + " TYPE=" + dt.upper() + " CGLOCK=0 FGLOCK=0 LOCKFREE=1"
                    os.system(make_cmd)
                    r_cmd = run_cmd + " >> " + result_path + temp_file
                    r_cmd = r_cmd + "_" + dt + "_tl" + str(t) + "_dpus" + str(r) +  "_lf.out" 
                    os.system(r_cmd)       



def main():
    input_path = sys.argv[1]
    run(input_path)

if __name__ == "__main__":
    main()
