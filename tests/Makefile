#
# tensorForth tests/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
TSRCS += \
	tests/t_tensor.cu

TOBJS += \
	./tests/t_tensor.o

# Additional CUDA includes if any
CU_INCS :=

# Cutlass includes
CL_INC  := ${CUTLASS_HOME}/include
CL_TOOL := ${CUTLASS_HOME}/tools

.PHONY: test clean clean-test

# Each subdirectory must supply rules for building sources it contributes
test: $(TOBJS) ./src/tensor.o
	@echo 'Building test case: $@'
	@echo 'Invoking: NVCC linker'
	${CUDA_HOME}/bin/nvcc -ccbin g++ --cudart=static -o "$(<:.o=)" \
		-L${CUDA_LIB} \
		-L${CUTLASS_HOME}/build/tools/library \
		-l${CL_LIB} \
		-gencode arch=${CUDA_ARCH},code=${CUDA_CODE} \
		$^
	@echo 'Finished building: $@'
	@echo ' '

tests/%.o: tests/%.cu $(INCS)
	@echo 'Building test file: $<'
	@echo 'Invoking: NVCC Compiler'
	${CUDA_HOME}/bin/nvcc -ccbin g++ \
		-I${CL_INC} -I${CL_TOOL}/library/include \
		-I${CL_TOOL}/util/include \
		-t=0 -c -std=c++14 -O3 -D__CUDACC__ -o "$@" "$<" \
		--device-c --extended-lambda --expt-relaxed-constexpr \
		--device-debug --debug \
		-gencode arch=${CUDA_ARCH},code=${CUDA_CODE} \
		--keep-dir ${APP_HOME}/tests
	@echo 'Finished building target: $@'
	@echo ' '

clean: clean-tst

clean-tst:
	-$(RM) $(TOBJS)
