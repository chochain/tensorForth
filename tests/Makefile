#
# tensorForth src/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
TSRCS += \
	./util.cpp \
	./t_tensor.cu

TOBJS += \
	./t_tensor.o

# Additional CUDA includes if any
CU_INCS :=

# Cutlass includes
CL_INC  := ${CUTLASS_HOME}/include
CL_TOOL := ${CUTLASS_HOME}/tools

.PHONY: clean clean-test

# Each subdirectory must supply rules for building sources it contributes
./tests/%.o: ./tests/%.cu ./tests/%.cpp
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
	${CUDA_HOME}/bin/nvcc -ccbin g++ --cudart=static -o "$*" \
		-L${CUDA_LIB} \
		-L${CUTLASS_HOME}/build/tools/library \
		-l${CL_LIB} \
		-gencode arch=${CUDA_ARCH},code=${CUDA_CODE} \
		$(OBJS) $(USER_OBJS) $(USER_LIBS)
	@echo 'Finished building: $<'
	@echo ' '

clean: clean-tst

clean-tst:
	-$(RM) $(TOBJS)

