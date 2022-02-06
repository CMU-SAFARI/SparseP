#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <getopt.h>
#include <unistd.h>
#include <string.h>

typedef struct Params {
    char* fileName;
    unsigned int vert_partitions;
    unsigned int nthreads;
} Params;

static void usage() {
    fprintf(stderr,
            "\nUsage:  ./program [options]"
            "\n"
            "\nOptions:"
            "\n    -h        help"
            "\n    -f <F>    Input matrix file name (default=roadNet-TX.mtx)"
            "\n    -v <V>    # of vertical partitions in the matrix (default=2)"
            "\n    -n <N>    # of OpenMP threads in merge step (default=1)"
            "\n");
}

static char *strremove(char *str, const char *sub) {
    size_t len = strlen(sub);
    if (len > 0) {
        char *p = str;
        while ((p = strstr(p, sub)) != NULL) {
            memmove(p, p + len, strlen(p + len) + 1);
        }
    }
    return str;
}

static struct Params input_params(int argc, char **argv) {
    struct Params p;

    // Set default input matrix
    char *rel_dir = "spmv/2D/DCOO";
    char *abs_dir = (char *) malloc(1024);
    abs_dir = getcwd(abs_dir, 1024);
    p.fileName =  strcat(strremove(abs_dir, rel_dir), (char *)"inputs/roadNet-TX.mtx");
    p.vert_partitions = 2;
    p.nthreads = 1;

    int opt;
    while((opt = getopt(argc, argv, "h:f:v:n:")) >= 0) {
        switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'f': p.fileName      = optarg; break;
            case 'v': p.vert_partitions= atoi(optarg); break;
            case 'n': p.nthreads      = atoi(optarg); break;
            default:
                      fprintf(stderr, "\nUnrecognized option!\n");
                      usage();
                      exit(0);
        }
    }

    free(abs_dir);
    return p;
}


#endif
