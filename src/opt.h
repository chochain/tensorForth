/*! @file
  @brief tensorForth command line options
*/
#ifndef TEN4_SRC_OPT_H_
#define TEN4_SRC_OPT_H_

#include <getopt.h>            ///< GNU option parser
///////////////////////////////////////////////////////////////////////////////////////////////////
// Command line options parsing
struct Options {
    int   verbose         = 0;
    int   device_id       = 0;
    bool  help            = false;
    float problem_size[3] = {1024, 512, 2048};
    float alpha           = 1.0;
    float beta            = 0.0;
    int   nbatch          = 1;
    int   iteration       = 1;
    ///
    /// command line option parser
    ///
    void parse(int argc, char **argv) {
        /*
        static struct option olist[] = {
            {"help",      no_argument,       0,                'h' },
            {"verbose",   required_argument, &verbose,         'v' },
            {"device",    required_argument, &device_id,       'd' },
            {"y",         required_argument, &problem_size[0], 'y' },
            {"x",         required_argument, &problem_size[1], 'x' },
            {"k",         required_argument, &problem_size[2], 'k' },
            {"nbatch",    required_argument, &nbatch,          'n' },
            {"iteration", required_argument, &iterations,      'i' },
            {"alpha",     required_argument, &alpha,           'a' },
            {"beta",      required_argument, &beta,            'b' }
        };
        */
        char opt;
        while ((opt = getopt(argc, argv, "hv:d:y:x:k:n:i:a:b:")) != -1) {
            switch (opt) {
            case 'h': help      = true;         break;
            case 'v': verbose   = atoi(optarg); break;
            case 'd': device_id = atoi(optarg); break;
            case 'y': problem_size[0] = atoi(optarg); break;
            case 'x': problem_size[1] = atoi(optarg); break;
            case 'k': problem_size[2] = atoi(optarg); break;
            case 'n': nbatch    = atoi(optarg); break;
            case 'i': iteration = atoi(optarg); break;
            case 'a': alpha     = atof(optarg); break;
            case 'b': beta      = atof(optarg); break;
            default:
                print_usage(std::cerr);
                exit(EXIT_FAILURE);
            }
        }
    }
    //
    // print device properties
    //
    int check_versions(cudaDeviceProp &props) {
        const char *err[] = {
            "Volta Tensor Core operations must be run on a machine with compute capability at least 70.",
            "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later.",
            "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later."
        };
        //
        // Volta Tensor Core operations are first available in CUDA 10.1 Toolkit.
        //
        // Turing Tensor Core operations are first available in CUDA 10.2 Toolkit.
        //
        if (props.major < 7) { std::cerr << err[0] << std::endl; return 0; }
        else if (props.major == 7 && props.minor <= 2) {
            //
            // If running on the Volta architecture, at least CUDA 10.1 Toolkit is required to run this example.
            //
            if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
                std::cerr << err[1] << std::endl; return 0;
            }
        }
        else if (props.major == 7 && props.minor >= 5) {
            //
            // If running on the Turing architecture, at least CUDA 10.2 Toolkit is required to run this example.
            //
            if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
                std::cerr << err[2] << std::endl; return 0;
            }
        }
        else {
            // NVIDIA Ampere Architecture GPUs (SM80 and later) are fully supported on CUDA 11 Toolkit and beyond.
            //
            // fall through
        }
        return 1;
    }
    std::ostream &show_device_prop(std::ostream &out, cudaDeviceProp &p) {
        const char *yes_no[] = { "No", "Yes" };
        out << "\tName:                          " << p.name << "\n"
            << "\tCUDA version:                  " << p.major << "." << p.minor << "\n"
            << "\tTotal global memory:           " << (U32)(p.totalGlobalMem>>20) << "M\n"
            << "\tTotal shared memory per block: " << (U32)(p.sharedMemPerBlock>>10) << "K\n"
            << "\tNumber of multiprocessors:     " << p.multiProcessorCount << "\n"
            << "\tTotal registers per block:     " <<  (p.regsPerBlock>>10) << "K\n"
            << "\tWarp size:                     " << p.warpSize << std::endl
            << "\tMax memory pitch:              " << (U32)(((U64)p.memPitch+1)>>20) << "M\n"
            << "\tMax threads per block:         " << p.maxThreadsPerBlock << "\n"
            << "\tMax dim of block:              [";
        for (int i = 0; i < 3; ++i)
            out << p.maxThreadsDim[i] << (i<2 ? ", " : "]\n");
        out << "\tMax dim of grid:               ["
            << (U32)(((U64)p.maxGridSize[0]+1)>>20) << "M, "
            << (U32)((p.maxGridSize[1]+1)>>10) << "K, "
            << (U32)((p.maxGridSize[2]+1)>>10) << "K]\n";
        out << "\tClock rate:                    " << p.clockRate/1000 << "KHz\n"
            << "\tTotal constant memory:         " << (U32)(p.totalConstMem>>10) << "K\n"
            << "\tTexture alignment:             " << p.textureAlignment << "\n"
            << "\tConcurrent copy and execution: " << yes_no[p.deviceOverlap] << "\n"
            << "\tKernel execution timeout:      " << yes_no[p.kernelExecTimeoutEnabled] << std::endl;
       return out;
    }
    int check_devices(std::ostream &out) {
        int n;
        cudaGetDeviceCount(&n);
        for (int device_id = 0; device_id < n; ++device_id) {
            printf("\nCUDA Device #%d\n", device_id);
            cudaDeviceProp props;
            cudaError_t    error = cudaGetDeviceProperties(&props, device_id);
            if (error != cudaSuccess) {
                std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
                continue;
            }
            check_versions(props);
            show_device_prop(out, props);
        }
        return n;
    }
    /// Prints the usage statement.
    std::ostream &print_usage(std::ostream &out) const {
        out << "\ntensorForth - Forth does tensors, in GPU\n"
            << "Options:\n"
            << "  -h        list all GPUs and this usage statement.\n"
            << "  -d <int>  GPU device id\n"
            << "  -v <int>  Verbosity level, 0: default, 1: mmu debug, 2: more details\n\n"
            << "  -y <int>  GEMM M dimension\n"
            << "  -x <int>  GEMM N dimension\n"
            << "  -k <int>  GEMM K dimension\n"
            << "  -i <int>  Number of profiling iterations to perform.\n"
            << "  -n <int>  Number of GEMM operations executed in one batch\n"
            << "  -a <f32>  Epilogue scalar alpha (real part)\n"
            << "  -b <f32>  Epilogue scalar alpha (imaginary part)\n\n"
            << "Examples:\n"
            << "$ ./tests/ten4 -h  ;# display help\n"
            << "$ ./tests/ten4 -d 0\n"
            << "$ ./tests/ten4 -n 7 -y 1024 -x 512 -k 2048 -a 2.0 -b 0.707\n";
        return out;
    }
};
#endif // TEN4_SRC_OPT_H_
