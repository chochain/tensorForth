#
# tensorForth src/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
SRCS +=

INCS +=

OBJS += \
	./src/util.o \
	./src/aio.o \
	./src/mmu.o \
	./src/eforth.o \
	./src/tensor.o \
	./src/tlsf.o \
	./src/ten4.o

# Additional CUDA includes if any
CU_INCS :=

# Cutlass includes
CL_INC  := ${CUTLASS_HOME}/include
CL_TOOL := ${CUTLASS_HOME}/tools

.PHONY: clean clean-src

# Each subdirectory must supply rules for building sources it contributes
src/%.o: src/%.cu src/%.h
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
