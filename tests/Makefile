#
# tensorForth tests/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
TSRCS += \
	tests/t_tensor.cu \
	tests/t_tlsf.cu \
	tests/t_bmpvu.cu

TOBJS += \
	./tests/t_tensor.o \
	./tests/t_tlsf.o \
	./tests/t_bmpvu.o

# Additional CUDA includes if any
CU_INCS := \
	${CUTLASS_HOME}/include \
	${CUTLASS_HOME}/tools/library/include \
	${CUTLASS_HOME}/tools/util/include \
	${CUDA_HOME}/cuda-samples/Common \
	${CUDA_HOME}/cuda-samples/Samples/2_Concepts_and_Techniques/imageDenoising

INC_LIST = $(CU_INCS:%=-I%)

# Cutlass includes
CU_BUILD:= ${CUDA_HOME}/bin/nvcc -ccbin g++ --cudart=static \
		-L$(CUDA_LIB) \
		-L${CUTLASS_HOME}/build/tools/library \
	    -lcutlass -lcusolver -lGL -lGLU -lglut \
		-gencode arch=${CUDA_ARCH},code=${CUDA_CODE}

.PHONY: test clean clean-test

test: t_tensor t_tlsf t_mmu_tensor t_inverse t_solver t_bmpvu

# Each subdirectory must supply rules for building sources it contributes
t_tensor: ./tests/t_tensor.o ./src/tensor.o ./src/util.o ./src/tlsf.o
	$(CU_BUILD) -o "$(<:.o=)" $^

t_tlsf: ./tests/t_tlsf.o ./src/tlsf.o
	$(CU_BUILD)	-o "$(<:.o=)" $^

t_mmu_tensor: ./tests/t_mmu_tensor.o ./src/tensor.o ./src/mmu.o ./src/tlsf.o ./src/util.o
	$(CU_BUILD)	-o "$(<:.o=)" $^

t_inverse: ./tests/t_inverse.o
	$(CU_BUILD)	-o "$(<:.o=)" $^

t_bmpvu: ./tests/t_bmpvu.o ./src/vu.o ./src/gui.o
	$(CU_BUILD)	-o "$(<:.o=)" $^

t_solver: ./tests/t_solver.o
	$(CU_BUILD)	-o "$(<:.o=)" $^

tests/%.o: tests/%.cu $(INCS)
	@echo 'Building test file: $<'
	@echo 'Invoking: NVCC Compiler'
	${CUDA_HOME}/bin/nvcc -ccbin g++ \
		${INC_LIST} \
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
