#
# tensorForth src/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
SRCS += \
    ./src/util.cu \
    ./src/aio.cu \
    ./src/mmu.cu \
    ./src/eforth.cu \
    ./src/tensor.cu \
    ./src/ten4.cu

OBJS += \
    ./src/util.o \
    ./src/aio.o \
    ./src/mmu.o \
    ./src/eforth.o \
    ./src/tensor.o \
    ./src/ten4.o

INCS += \
    ./src/ten4_config.h \
    ./src/ten4_types.h \
    ./src/tensor.h \
    ./src/util.h \
    ./src/aio.h \
    ./src/istream.h \
    ./src/ostream.h \
    ./src/mmu.h \
    ./src/eforth.h \
    ./src/opt.h \
    ./src/ten4.h

# Additional CUDA includes if any
CU_INCS :=

# Cutlass includes
CL_INC  := ${CUTLASS_HOME}/include
CL_TOOL := ${CUTLASS_HOME}/tools

# Each subdirectory must supply rules for building sources it contributes
./src/%.o: ./src/%.cu $(INCS)
	@echo 'Building source file: $<'
	@echo 'Invoking: NVCC Compiler'
	${CUDA_HOME}/bin/nvcc -ccbin g++ \
        -I$(CL_INC) \
        -I$(CL_TOOL)/library/include \
        -I$(CL_TOOL)/util/include \
        -t=0 -c -std=c++14 -O3 -D__CUDACC__ -o "$@" "$<" \
        --device-c --extended-lambda --expt-relaxed-constexpr \
        --device-debug --debug \
         -gencode arch=${CUDA_ARCH},code=${CUDA_CODE} \
        --keep-dir $(APP_HOME)/src
	@echo 'Finished building: $<'
	@echo ' '

clean: clean-src

clean-src:
	-$(RM) $(OBJS)

.PHONY: clean-src

