#
# tensorForth tests/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
TSRCS += \
	tests/t_tensor.cu \
	tests/t_tlsf.cu


TOBJS += \
	./tests/t_tensor.o \
	./tests/t_tlsf.o

# Additional CUDA includes if any
CU_INCS :=

# Cutlass includes
CL_INC  := ${CUTLASS_HOME}/include
CL_TOOL := ${CUTLASS_HOME}/tools
CU_BUILD:= ${CUDA_HOME}/bin/nvcc -ccbin g++ --cudart=static \
		-L$(CUDA_LIB) \
		-L${CUTLASS_HOME}/build/tools/library \
		-lcutlass \
		-gencode arch=${CUDA_ARCH},code=${CUDA_CODE}

.PHONY: test clean clean-test

test: t_tensor t_tlsf t_mmu_tensor t_inverse

# Each subdirectory must supply rules for building sources it contributes
t_tensor: ./tests/t_tensor.o ./src/tensor.o
	$(CU_BUILD) -o "$(<:.o=)" $^

t_tlsf: ./tests/t_tlsf.o ./src/tlsf.o
	$(CU_BUILD)	-o "$(<:.o=)" $^

t_mmu_tensor: ./tests/t_mmu_tensor.o ./src/tensor.o ./src/mmu.o ./src/tlsf.o
	$(CU_BUILD)	-o "$(<:.o=)" $^

t_inverse: ./tests/t_inverse.o
	$(CU_BUILD)	-o "$(<:.o=)" $^

tests/%.o: tests/%.cu $(INCS)
	@echo 'Building test file: $<'
	@echo 'Invoking: NVCC Compiler'
	${CUDA_HOME}/bin/nvcc -ccbin g++ \
		-I${CL_INC} -I${CL_TOOL}/library/include \
		-I${CL_TOOL}/util/include \
		-D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING \
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
