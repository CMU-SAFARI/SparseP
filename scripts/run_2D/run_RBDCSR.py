import os 
import sys
import glob
import getpass

result_path = "results/"

def run(input_path):

    NR_TASKLETS = [16]
    NR_DPUS = [1024, 2048]
    NR_VERT_PARTITIONS = [4, 8]
    DATATYPES = ["int8", "int16", "int32", "int64"]
    MATRICES = ["ldoor.mtx", "af_shell1.mtx", "roadNet-TX.mtx", "parabolic_fem.mtx", "poisson3Db.mtx", "delaunay_n19.mtx", "com-Youtube.mtx", "pkustk14.mtx"]

    pwd = os.getcwd()
    path = pwd.replace("scripts/run_2D", "") + "spmv/2D/RBDCSR"
    os.chdir(path)
    os.makedirs(result_path, exist_ok=True)
    for file in MATRICES:
        print(file)
        run_cmd = "bin/spmv_host " + "-n 16 -f " + input_path + file
        for dt in DATATYPES:
            for r in NR_DPUS:
                for t in NR_TASKLETS:
                    for c in NR_VERT_PARTITIONS:
                        # Balance NNZs across tasklets - Fine-grained data transfers (default)
                        os.system("make clean")
                        make_cmd = "make NR_DPUS=" + str(r) + " NR_TASKLETS=" + str(t) + " TYPE=" + dt.upper() 
                        os.system(make_cmd)
                        temp_file = str(file[:-4]) 
                        temp_file = temp_file.replace("_", "-") 
                        r_cmd = run_cmd + " -v " + str(c)
                        r_cmd = r_cmd + " >> " + result_path + temp_file
                        r_cmd = r_cmd + "_" + dt + "_tl" + str(t) + "_dpus" + str(r) + "_vps" + str(c) +  ".out" 
                        os.system(r_cmd)



def main():
    input_path = sys.argv[1]
    run(input_path)

if __name__ == "__main__":
    main()
