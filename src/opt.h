/*! @file
  @brief tensorForth command line options
*/
#ifndef TEN4_SRC_OPT_H_
#define TEN4_SRC_OPT_H_
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Result structure
struct Result {
    double          runtime_ms;
    double          gflops;
    cutlass::Status status;
    cudaError_t     error;
    bool            passed;
    //
    // Methods
    //
    Result(
        double          runtime_ms = 0,
        double          gflops     = 0,
        cutlass::Status status     = cutlass::Status::kSuccess,
        cudaError_t     error      = cudaSuccess
        ):
        runtime_ms(runtime_ms),
        gflops(gflops),
        status(status),
        error(error),
        passed(true) { }
};
///////////////////////////////////////////////////////////////////////////////////////////////////
// Command line options parsing
struct Options {
    bool help;
    cutlass::gemm::GemmCoord problem_size;
    cutlass::complex<float>  alpha;
    cutlass::complex<float>  beta;
    int  batch_count;
    bool reference_check;
    int  iterations;
  
    Options():
        help(false),
        problem_size({1024, 1024, 1024}),
        batch_count(1),
        reference_check(true),
        iterations(20),
        alpha(1),
        beta(0) { }

    bool valid() { return true; }

    // Parses the command line
    void parse(int argc, char const **args) {
        cutlass::CommandLine cmd(argc, args);

        if (cmd.check_cmd_line_flag("help")) { help = true; }
        
        cmd.get_cmd_line_argument("m",       problem_size.m());
        cmd.get_cmd_line_argument("n",       problem_size.n());
        cmd.get_cmd_line_argument("k",       problem_size.k());
        cmd.get_cmd_line_argument("batch",   batch_count);

        cmd.get_cmd_line_argument("alpha",   alpha.real());
        cmd.get_cmd_line_argument("alpha_i", alpha.imag());
        cmd.get_cmd_line_argument("beta",    beta.real());
        cmd.get_cmd_line_argument("beta_i",  beta.imag());
    
        cmd.get_cmd_line_argument("iterations", iterations);
    }

    /// Prints the usage statement.
    std::ostream &print_usage(std::ostream &out) const {
        out << "10_planar_complex example\n\n"
            << "  This example uses the CUTLASS Library to execute Planar Complex GEMM computations.\n\n"
            << "Options:\n\n"
            << "  --help                      If specified, displays this usage statement.\n\n"
            << "  --m=<int>                   GEMM M dimension\n"
            << "  --n=<int>                   GEMM N dimension\n"
            << "  --k=<int>                   GEMM K dimension\n"
            << "  --batch=<int>               Number of GEMM operations executed in one batch\n"
            << "  --alpha=<f32>               Epilogue scalar alpha (real part)\n"
            << "  --alpha_i=<f32>             Epilogue scalar alpha (imaginary part)\n"
            << "  --beta=<f32>                Epilogue scalar beta (real part)\n\n"
            << "  --beta_i=<f32>              Epilogue scalar beta (imaginary part)\n\n"
            << "  --iterations=<int>          Number of profiling iterations to perform.\n\n"
            << "\n\nExamples:\n\n"
            << "$ ./examples/10_planar_complex/10_planar_complex  --batch=7 --m=1024 --n=512 --k=1024 \\\n"
            << "     --alpha=2 --alpha_i=-2 --beta=0.707 --beta_i=-.707\n\n";
        return out;
    }
    /// Compute performance in GFLOP/s
    double gflops(double runtime_s) const {
        // Number of real-valued multiply-adds 
        int64_t fmas = problem_size.product() * batch_count * 4;
    
        // Two flops per multiply-add
        return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
    }
};
#endif // TEN4_SRC_OPT_H_
