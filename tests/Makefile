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
	t_gl \
	t_imgvu

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
	${CUTLASS_HOME}/tools/util/include \
    /u01/src/stb

INC_LST = $(CU_INCS:%=-I%)

GL_LIB = -lGL -lGLU -lglut -lX11

# Cutlass includes
CU_BUILD:= \

.PHONY: test clean clean-test

test: $(TTGTS)

# Each subdirectory must supply rules for building sources it contributes
$(TTGTS): $(TTGTS:%=tests/%.o) 
	@echo '<Test><Action>Link</Action><Filename>$@</Filename>'
	${CUDA_HOME}/bin/nvcc -ccbin g++ --cudart=static \
		-L$(CUDA_LIB) \
		-L${CUTLASS_HOME}/build/tools/library \
		-lcutlass -lcusolver $(GL_LIB) \
		-gencode arch=${CUDA_ARCH},code=${CUDA_CODE} \
		-o "./tests/$(@:.o=)" "./tests/$@.o" $(TOBJS)
	@echo '<Status>OK</Status></Test>'
	@echo ' '

tests/%.o: tests/%.cu $(TOBJS) $(CU_INCS)
	@echo '<Test><Action>Compile></Action><Filename>$<</Filename>'
	${CUDA_HOME}/bin/nvcc -ccbin g++ \
		${INC_LST} \
		-D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING \
		-t=0 -c -std=c++14 -O3 -D__CUDACC__ -o "$@" "$<" \
		--device-c --extended-lambda --expt-relaxed-constexpr \
		--device-debug --debug \
		-gencode arch=${CUDA_ARCH},code=${CUDA_CODE} \
		--keep-dir ${APP_HOME}/tests
	@echo '<Status>OK</Status></Test>'
	@echo ' '

clean: clean-tst

clean-tst:
	-$(RM) $(TTGTS:%=./tests/%.o)
