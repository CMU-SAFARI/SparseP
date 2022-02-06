#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <getopt.h>
#include <unistd.h>
#include <string.h>

typedef struct Params {
    char* fileName;
} Params;

static void usage() {
    fprintf(stderr,
            "\nUsage:  ./program [options]"
            "\n"
            "\nOptions:"
            "\n    -h        help"
            "\n    -f <F>    Input matrix file name (default=roadNet-TX.mtx)"
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
    char *rel_dir = "spmv/1D/COO-nnz";
    char *abs_dir = (char *) malloc(1024);
    abs_dir = getcwd(abs_dir, 1024);
    p.fileName =  strcat(strremove(abs_dir, rel_dir), (char *)"inputs/roadNet-TX.mtx");

    int opt;
    while((opt = getopt(argc, argv, "h:f:")) >= 0) {
        switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'f': p.fileName      = optarg; break;
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
