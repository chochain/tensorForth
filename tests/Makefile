#
# tensorForth tests/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
TTGTS := \
	t_tensor \
	t_tlsf \
	t_mmu_tensor \
	t_inverse \
	t_bmpvu	\
	t_solver \
	t_gl

TOBJS := \
	./src/util.o \
	./src/tlsf.o \
	./src/mmu.o \
	./src/tensor.o \
	./src/vu.o \
	./src/gui.o

# Additional CUDA includes if any
CU_INCS := \
	${CUTLASS_HOME}/include \
	${CUTLASS_HOME}/tools/library/include \
	${CUTLASS_HOME}/tools/util/include

INC_LST = $(CU_INCS:%=-I%)

GL_LIB = -lGL -lGLU -lglut -lX11

# Cutlass includes
CU_BUILD:= ${CUDA_HOME}/bin/nvcc -ccbin g++ --cudart=static \
		-L$(CUDA_LIB) \
		-L${CUTLASS_HOME}/build/tools/library \
	    -lcutlass -lcusolver $(GL_LIB) \
		-gencode arch=${CUDA_ARCH},code=${CUDA_CODE}

.PHONY: test clean clean-test

test: $(TTGTS)

# Each subdirectory must supply rules for building sources it contributes
t_tensor: ./tests/t_tensor.o ./src/tensor.o ./src/util.o
	$(CU_BUILD) -o "$(<:.o=)" $^

t_tlsf: ./tests/t_tlsf.o ./src/tlsf.o
	$(CU_BUILD)	-o "$(<:.o=)" $^

t_mmu_tensor: ./tests/t_mmu_tensor.o $(TOBJS)
	$(CU_BUILD)	-o "$(<:.o=)" $^

t_inverse: ./tests/t_inverse.o
	$(CU_BUILD)	-o "$(<:.o=)" $^

t_bmpvu: ./tests/t_bmpvu.o $(TOBJS)
	$(CU_BUILD)	-o "$(<:.o=)" $^

t_solver: ./tests/t_solver.o
	$(CU_BUILD)	-o "$(<:.o=)" $^

t_gl: ./tests/t_gl.o
	$(CU_BUILD)	-o "$(<:.o=)" $^

tests/%.o: tests/%.cu $(CU_INCS)
	@echo 'Building test file: $<'
	@echo 'Invoking: NVCC Compiler'
	${CUDA_HOME}/bin/nvcc -ccbin g++ \
		${INC_LST} \
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
	-$(RM) $(TTGTS:%=./tests/%.o)
