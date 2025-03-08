#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

using namespace std;

void check(cudaError_t err, char const* func, char const* file, int line) {
    if (err != cudaSuccess) {
        cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << endl;
        cerr << cudaGetErrorString(err) << " " << func << endl;
        exit(EXIT_FAILURE);
    }
}

void check_last(char const* file, int line) {
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess) {
        cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << endl;
        cerr << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

typedef float DU;
#define CHK_ERR(val) check((val), #val, __FILE__, __LINE__)
#define LAST_ERR()   check_last(__FILE__, __LINE__)
#define ASSERT(f,s)  static_assert(f,s)

template <class T>
DU measure_performance(
    function<T(cudaStream_t)> bound_function,
    cudaStream_t stream, size_t num_repeats = 10,
    size_t num_warmups = 10
    ) {
    cudaEvent_t start, stop;
    DU time;
    
    CHK_ERR(cudaEventCreate(&start));
    CHK_ERR(cudaEventCreate(&stop));

    for (size_t i{0}; i < num_warmups; ++i) {
        bound_function(stream);
    }

    CHK_ERR(cudaStreamSynchronize(stream));

    CHK_ERR(cudaEventRecord(start, stream));
    for (size_t i{0}; i < num_repeats; ++i) {
        bound_function(stream);
    }
    CHK_ERR(cudaEventRecord(stop, stream));
    CHK_ERR(cudaEventSynchronize(stop));
    LAST_ERR();
    CHK_ERR(cudaEventElapsedTime(&time, start, stop));
    CHK_ERR(cudaEventDestroy(start));
    CHK_ERR(cudaEventDestroy(stop));

    DU const latency{time / num_repeats};

    return latency;
}

string std_string_centered(
    string const& s, size_t width,  char pad = ' '
    ) {
    size_t const l { s.length() };
    // Throw an exception if width is too small.
    if (width < l) {
        throw runtime_error("Width is too small.");
    }
    size_t const left_pad { (width - l) / 2 };
    size_t const right_pad { width - l - left_pad };
    string const s_centered {
        string(left_pad, pad) + s + string(right_pad, pad) };
    return s_centered;
}

template <size_t NUM_THREADS>
__device__ DU
thread_block_reduce_sum(
    cooperative_groups::thread_block_tile<NUM_THREADS> group,
    DU shared_data[NUM_THREADS], DU val
    ) {
    ASSERT(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
    size_t thread_idx { group.thread_rank() };
    shared_data[thread_idx] = val;
    group.sync();
#pragma unroll
    for (size_t offset{group.size() / 2}; offset > 0; offset /= 2) {
        if (thread_idx < offset) {
            shared_data[thread_idx] += shared_data[thread_idx + offset];
        }
        group.sync();
    }
    // There will be no shared memory bank conflicts here.
    // Because multiple threads in a warp address the same shared memory
    // location, resulting in a broadcast.
    return shared_data[0];
}

__device__ DU
thread_block_reduce_sum(
    cooperative_groups::thread_block group,
    DU* shared_data,
    DU val
    ) {
    size_t const thread_idx { group.thread_rank() };
    shared_data[thread_idx] = val;
    group.sync();
    for (size_t stride{group.size() / 2}; stride > 0; stride /= 2) {
        if (thread_idx < stride) {
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        }
        group.sync();
    }
    return shared_data[0];
}

template <size_t NUM_WARPS>
__device__ DU
thread_block_reduce_sum(DU shared_data[NUM_WARPS])
{
    DU sum{0.0f};
#pragma unroll
    for (size_t i{0}; i < NUM_WARPS; ++i)
    {
        // There will be no shared memory bank conflicts here.
        // Because multiple threads in a warp address the same shared memory
        // location, resulting in a broadcast.
        sum += shared_data[i];
    }
    return sum;
}

__device__ DU
thread_reduce_sum(
    DU const* __restrict__ input_data,
    size_t start_offset,
    size_t num_elements,
    size_t stride
    ) {
    DU sum{0.0f};
    for (size_t i{start_offset}; i < num_elements; i += stride)
    {
        sum += input_data[i];
    }
    return sum;
}

__device__ DU
warp_reduce_sum(
    cooperative_groups::thread_block_tile<32> group,
    DU val
    ) {
#pragma unroll
    for (size_t offset{group.size() / 2}; offset > 0; offset /= 2) {
        // The shfl_down function is a warp shuffle operation that only exists
        // for thread block tiles of size 32.
        val += group.shfl_down(val, offset);
    }
    // Only the first thread in the warp will return the correct result.
    return val;
}

template <size_t NUM_THREADS>
__device__ DU
thread_block_reduce_sum_v1(
    DU const* __restrict__ input_data,
    size_t num_elements
    ) {
    ASSERT(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
    __shared__ DU shared_data[NUM_THREADS];
    size_t const thread_idx {
        cooperative_groups::this_thread_block().thread_index().x };
    DU sum {
        thread_reduce_sum(input_data, thread_idx, num_elements, NUM_THREADS) };
    shared_data[thread_idx] = sum;
    // This somehow does not work.
    // static thread block cooperative groups is still not supported.
    // cooperative_groups::thread_block_tile<NUM_THREADS> const
    // thread_block{cooperative_groups::tiled_partition<NUM_THREADS>(cooperative_groups::this_thread_block())};
    // DU const block_sum{thread_block_reduce_sum<NUM_THREADS>(thread_block,
    // shared_data, sum)}; This works.
    DU const block_sum { thread_block_reduce_sum(
        cooperative_groups::this_thread_block(), shared_data, sum) };
    return block_sum;
}

template <size_t NUM_THREADS, size_t NUM_WARPS = NUM_THREADS / 32>
__device__ DU
thread_block_reduce_sum_v2(
    DU const* __restrict__ input_data,
    size_t num_elements
    ) {
    ASSERT(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
    __shared__ DU shared_data[NUM_WARPS];
    
    size_t const thread_idx { cooperative_groups::this_thread_block().thread_index().x };
    DU sum { thread_reduce_sum(input_data, thread_idx, num_elements, NUM_THREADS) };
    
    cooperative_groups::thread_block_tile<32>
    const warp{ cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block()) };
    
    sum = warp_reduce_sum(warp, sum);
    if (warp.thread_rank() == 0) {
        shared_data[cooperative_groups::this_thread_block().thread_rank() / 32] = sum;
    }
    cooperative_groups::this_thread_block().sync();
    
    DU const block_sum { thread_block_reduce_sum<NUM_WARPS>(shared_data) };
    return block_sum;
}

template <size_t NUM_THREADS>
__global__ void
batched_reduce_sum_v1(
    DU* __restrict__ output_data,
    DU const* __restrict__ input_data,
    size_t num_elements_per_batch
    ) {
    ASSERT(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
    size_t const block_idx  { cooperative_groups::this_grid().block_rank() };
    size_t const thread_idx { cooperative_groups::this_thread_block().thread_rank() };
    DU  const block_sum  {
        thread_block_reduce_sum_v1<NUM_THREADS>(
            input_data + block_idx * num_elements_per_batch, num_elements_per_batch) };
    if (thread_idx == 0) {
        output_data[block_idx] = block_sum;
    }
}

template <size_t NUM_THREADS>
__global__ void
batched_reduce_sum_v2(
    DU* __restrict__ output_data,
    DU const* __restrict__ input_data,
    size_t num_elements_per_batch
    ) {
    ASSERT(NUM_THREADS % 32 == 0,"NUM_THREADS must be a multiple of 32");
    constexpr size_t NUM_WARPS { NUM_THREADS / 32 };
    size_t const block_idx  { cooperative_groups::this_grid().block_rank() };
    size_t const thread_idx { cooperative_groups::this_thread_block().thread_rank() };
    DU  const block_sum  {
        thread_block_reduce_sum_v2<NUM_THREADS, NUM_WARPS>(
            input_data + block_idx * num_elements_per_batch, num_elements_per_batch) };
    if (thread_idx == 0) {
        output_data[block_idx] = block_sum;
    }
}

template <size_t NUM_THREADS, size_t NUM_BLOCK_ELEMENTS>
__global__ void
full_reduce_sum(
    DU* output,
    DU const* __restrict__ input_data,
    size_t num_elements,
    DU* workspace
    ) {
    ASSERT(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
    ASSERT(NUM_BLOCK_ELEMENTS % NUM_THREADS == 0,
                  "NUM_BLOCK_ELEMENTS must be a multiple of NUM_THREADS");
    // Workspace size: num_elements.
    size_t const num_grid_elements {
        NUM_BLOCK_ELEMENTS * cooperative_groups::this_grid().num_blocks() };
    DU* const workspace_ptr_1 { workspace };
    DU* const workspace_ptr_2 { workspace + num_elements / 2 };
    size_t remaining_elements { num_elements };

    // The first iteration of the reduction.
    DU* workspace_output_data {workspace_ptr_1 };
    size_t const num_grid_iterations {
        (remaining_elements + num_grid_elements - 1) / num_grid_elements };
    for (size_t i{0}; i < num_grid_iterations; ++i) {
        size_t const grid_offset { i * num_grid_elements };
        size_t const block_offset {
            grid_offset +
                cooperative_groups::this_grid().block_rank() * NUM_BLOCK_ELEMENTS };
        size_t const num_actual_elements_to_reduce_per_block {
            remaining_elements >= block_offset
                ? min(NUM_BLOCK_ELEMENTS, remaining_elements - block_offset)
                : 0 };
        DU const block_sum {
            thread_block_reduce_sum_v1<NUM_THREADS>(
                input_data + block_offset,
                num_actual_elements_to_reduce_per_block) };
        if (cooperative_groups::this_thread_block().thread_rank() == 0) {
            workspace_output_data
                [ i * cooperative_groups::this_grid().num_blocks() +
                  cooperative_groups::this_grid().block_rank() ] = block_sum;
        }
    }
    cooperative_groups::this_grid().sync();
    remaining_elements =
        (remaining_elements + NUM_BLOCK_ELEMENTS - 1) / NUM_BLOCK_ELEMENTS;

    // The rest iterations of the reduction.
    DU* workspace_input_data {workspace_output_data };
    workspace_output_data = workspace_ptr_2;
    while (remaining_elements > 1) {
        size_t const num_grid_iterations {
            (remaining_elements + num_grid_elements - 1) / num_grid_elements };
        for (size_t i{0}; i < num_grid_iterations; ++i) {
            size_t const grid_offset {i * num_grid_elements };
            size_t const block_offset {
                grid_offset +
                    cooperative_groups::this_grid().block_rank() * NUM_BLOCK_ELEMENTS };
            size_t const num_actual_elements_to_reduce_per_block{
                remaining_elements >= block_offset
                    ? min(NUM_BLOCK_ELEMENTS, remaining_elements - block_offset)
                    : 0};
            DU const block_sum {
                thread_block_reduce_sum_v1<NUM_THREADS>(
                    workspace_input_data + block_offset,
                    num_actual_elements_to_reduce_per_block) };
            if (cooperative_groups::this_thread_block().thread_rank() == 0) {
                workspace_output_data
                    [i * cooperative_groups::this_grid().num_blocks() +
                     cooperative_groups::this_grid().block_rank()] = block_sum;
            }
        }
        cooperative_groups::this_grid().sync();
        remaining_elements =
            (remaining_elements + NUM_BLOCK_ELEMENTS - 1) / NUM_BLOCK_ELEMENTS;

        // Swap the input and output data.
        DU* const temp{workspace_input_data};
        workspace_input_data = workspace_output_data;
        workspace_output_data = temp;
    }

    // Copy the final result to the output.
    workspace_output_data = workspace_input_data;
    if (cooperative_groups::this_grid().thread_rank() == 0) {
        *output = workspace_output_data[0];
    }
}

template <size_t NUM_THREADS>
__host__ void
launch_batched_reduce_sum_v1(
    DU* output_data, DU const* input_data,
    size_t batch_size,
    size_t num_elements_per_batch,
    cudaStream_t stream
    ) {
    size_t const num_blocks { batch_size };
    batched_reduce_sum_v1<NUM_THREADS><<<num_blocks, NUM_THREADS, 0, stream>>>(
        output_data, input_data, num_elements_per_batch);
    LAST_ERR();
}

template <size_t NUM_THREADS>
__host__ void
launch_batched_reduce_sum_v2(
    DU* output_data,
    DU const* input_data,
    size_t batch_size,
    size_t num_elements_per_batch,
    cudaStream_t stream
    ) {
    size_t const num_blocks{batch_size};
    batched_reduce_sum_v2<NUM_THREADS><<<num_blocks, NUM_THREADS, 0, stream>>>(
        output_data, input_data, num_elements_per_batch);
    LAST_ERR();
}

template <size_t NUM_THREADS, size_t NUM_BLOCK_ELEMENTS>
__host__ void
launch_full_reduce_sum(
    DU* output,
    DU const* input_data,
    size_t num_elements,
    DU* workspace,
    cudaStream_t stream
    ) {
    // https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#grid-synchronization
    void const* func {
        reinterpret_cast<void const*>(full_reduce_sum<NUM_THREADS, NUM_BLOCK_ELEMENTS>) };
    int dev{0};
    cudaDeviceProp deviceProp;
    CHK_ERR(cudaGetDeviceProperties(&deviceProp, dev));
    dim3 const grid_dim {
        static_cast<unsigned int>(deviceProp.multiProcessorCount) };
    dim3 const block_dim{NUM_THREADS};

    // This will launch a grid that can maximally fill the GPU, on the
    // default stream with kernel arguments.
    // In practice, it's not always the best.
    // void const* func{reinterpret_cast<void const*>(
    //     full_reduce_sum<NUM_THREADS, NUM_BLOCK_ELEMENTS>)};
    // int dev{0};
    // dim3 const block_dim{NUM_THREADS};
    // int num_blocks_per_sm{0};
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, dev);
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, func,
    //                                               NUM_THREADS, 0);
    // dim3 const grid_dim{static_cast<unsigned int>(num_blocks_per_sm)};

    void* args[]{
        static_cast<void*>(&output),
        static_cast<void*>(&input_data),
        static_cast<void*>(&num_elements),
        static_cast<void*>(&workspace) };
    CHK_ERR(cudaLaunchCooperativeKernel(
                func, grid_dim, block_dim, args, 0, stream));
    LAST_ERR();
}

__host__ DU
profile_full_reduce_sum(
    function<void(DU*, DU const*, size_t, DU*, cudaStream_t)>
    full_reduce_sum_launch_function,
    size_t num_elements
    ) {
    cudaStream_t stream;
    CHK_ERR(cudaStreamCreate(&stream));

    constexpr DU element_value {1.0f};
    vector<DU> input_data(num_elements, element_value);
    DU output {0.0f};

    DU* d_input_data;
    DU* d_workspace;
    DU* d_output;

    CHK_ERR(cudaMalloc(&d_input_data, num_elements * sizeof(DU)));
    CHK_ERR(cudaMalloc(&d_workspace, num_elements * sizeof(DU)));
    CHK_ERR(cudaMalloc(&d_output, sizeof(DU)));

    CHK_ERR(
        cudaMemcpy(
            d_input_data, input_data.data(),
            num_elements * sizeof(DU),
            cudaMemcpyHostToDevice));

    full_reduce_sum_launch_function(
        d_output, d_input_data, num_elements, d_workspace, stream);
    CHK_ERR(cudaStreamSynchronize(stream));

    // Verify the correctness of the kernel.
    CHK_ERR(
        cudaMemcpy(
            &output, d_output, sizeof(DU), cudaMemcpyDeviceToHost));
    if (output != num_elements * element_value) {
        cout << "Expected: " << num_elements * element_value
                  << " but got: " << output << endl;
        throw runtime_error("Error: incorrect sum");
    }
    function<void(cudaStream_t)> const bound_function {
        bind(
            full_reduce_sum_launch_function,
            d_output, d_input_data, num_elements,
            d_workspace, placeholders::_1) };
    DU const latency { measure_performance<void>(bound_function, stream) };
    cout << "Latency: " << latency << " ms" << endl;

    // Compute effective bandwidth.
    size_t num_bytes {num_elements * sizeof(DU) + sizeof(DU) };
    DU const bandwidth {(num_bytes * 1e-6f) / latency };
    cout << "Effective Bandwidth: " << bandwidth << " GB/s" << endl;

    CHK_ERR(cudaFree(d_input_data));
    CHK_ERR(cudaFree(d_workspace));
    CHK_ERR(cudaFree(d_output));
    CHK_ERR(cudaStreamDestroy(stream));

    return latency;
}

__host__ DU
profile_batched_reduce_sum(
    function<void(DU*, DU const*, size_t, size_t, cudaStream_t)>
    batched_reduce_sum_launch_function,
    size_t batch_size,
    size_t num_elements_per_batch
    ) {
    size_t const num_elements{batch_size * num_elements_per_batch};

    cudaStream_t stream;
    CHK_ERR(cudaStreamCreate(&stream));

    constexpr DU element_value{1.0f};
    vector<DU> input_data(num_elements, element_value);
    vector<DU> output_data(batch_size, 0.0f);

    DU* d_input_data;
    DU* d_output_data;

    CHK_ERR(cudaMalloc(&d_input_data, num_elements * sizeof(DU)));
    CHK_ERR(cudaMalloc(&d_output_data, batch_size * sizeof(DU)));

    CHK_ERR(cudaMemcpy(d_input_data, input_data.data(),
                                num_elements * sizeof(DU),
                                cudaMemcpyHostToDevice));

    batched_reduce_sum_launch_function(d_output_data, d_input_data, batch_size,
                                       num_elements_per_batch, stream);
    CHK_ERR(cudaStreamSynchronize(stream));

    // Verify the correctness of the kernel.
    CHK_ERR(cudaMemcpy(output_data.data(), d_output_data,
                                batch_size * sizeof(DU),
                                cudaMemcpyDeviceToHost));
    for (size_t i{0}; i < batch_size; ++i)
    {
        if (output_data.at(i) != num_elements_per_batch * element_value)
        {
            cout << "Expected: " << num_elements_per_batch * element_value
                      << " but got: " << output_data.at(i) << endl;
            throw runtime_error("Error: incorrect sum");
        }
    }
    function<void(cudaStream_t)> const bound_function{bind(
        batched_reduce_sum_launch_function, d_output_data, d_input_data,
        batch_size, num_elements_per_batch, placeholders::_1)};
    DU const latency{measure_performance<void>(bound_function, stream)};
    cout << "Latency: " << latency << " ms" << endl;

    // Compute effective bandwidth.
    size_t num_bytes{num_elements * sizeof(DU) + batch_size * sizeof(DU)};
    DU const bandwidth{(num_bytes * 1e-6f) / latency};
    cout << "Effective Bandwidth: " << bandwidth << " GB/s" << endl;

    CHK_ERR(cudaFree(d_input_data));
    CHK_ERR(cudaFree(d_output_data));
    CHK_ERR(cudaStreamDestroy(stream));

    return latency;
}

int main()
{
    size_t const batch_size{64};
    size_t const num_elements_per_batch{1024 * 1024};

    constexpr size_t string_width{50U};
    cout << std_string_centered("", string_width, '~') << endl;
    cout << std_string_centered("NVIDIA GPU Device Info", string_width,
                                     ' ')
              << endl;
    cout << std_string_centered("", string_width, '~') << endl;

    // Query deive name and peak memory bandwidth.
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    cout << "Device Name: " << device_prop.name << endl;
    DU const memory_size {
        static_cast<DU>(device_prop.totalGlobalMem) / (1 << 30) };
    cout << "Memory Size: " << memory_size << " GB" << endl;
    DU const peak_bandwidth{
        static_cast<DU>(
            2.0f * device_prop.memoryClockRate * (device_prop.memoryBusWidth / 8) / 1.0e6)};
    cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << endl;

    cout << std_string_centered("", string_width, '~') << endl;
    cout << std_string_centered("Reduce Sum Profiling", string_width, ' ')
              << endl;
    cout << std_string_centered("", string_width, '~') << endl;

    cout << std_string_centered("", string_width, '=') << endl;
    cout << "Batch Size: " << batch_size << endl;
    cout << "Number of Elements Per Batch: " << num_elements_per_batch
              << endl;
    cout << std_string_centered("", string_width, '=') << endl;

    constexpr size_t NUM_THREADS_PER_BATCH {512};
    ASSERT(NUM_THREADS_PER_BATCH % 32 == 0,
                  "NUM_THREADS_PER_BATCH must be a multiple of 32");
    ASSERT(NUM_THREADS_PER_BATCH <= 1024,
                  "NUM_THREADS_PER_BATCH must be less than or equal to 1024");

    cout << "Batched Reduce Sum V1" << endl;
    DU const latency_v1 {
        profile_batched_reduce_sum(
            launch_batched_reduce_sum_v1<NUM_THREADS_PER_BATCH>,
            batch_size,
            num_elements_per_batch) };
    cout << std_string_centered("", string_width, '-') << endl;

    cout << "Batched Reduce Sum V2" << endl;
    DU const latency_v2 {
        profile_batched_reduce_sum(
            launch_batched_reduce_sum_v2<NUM_THREADS_PER_BATCH>,
            batch_size,
            num_elements_per_batch) };
    cout << std_string_centered("", string_width, '-') << endl;

    cout << "Full Reduce Sum" << endl;
    constexpr size_t NUM_THREADS{256};
    constexpr size_t NUM_BLOCK_ELEMENTS{NUM_THREADS * 1024};
    DU const latency_v3 {
        profile_full_reduce_sum(
            launch_full_reduce_sum<NUM_THREADS, NUM_BLOCK_ELEMENTS>,
            batch_size * num_elements_per_batch) };
    cout << std_string_centered("", string_width, '-') << endl;
}
