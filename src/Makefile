#
# tensorForth src/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
SRCS +=

INCS += \
	${CUDA_HOME}/cuda-samples/Common \
	${CUDA_HOME}/cuda-samples/Samples/2_Concepts_and_Techniques/imageDenoising

INC_PARMS = $(INCS:%=-I%)

OBJS += \
	./src/util.o \
	./src/aio.o \
	./src/mmu.o \
	./src/vm.o \
	./src/eforth.o \
	./src/tlsf.o \
	./src/tensor.o \
	./src/tenvm.o \
	./src/model.o \
	./src/model_fwd.o \
	./src/model_back.o \
	./src/netvm.o \
	./src/ten4.o \
	./src/imgvu_bmp.o \
    ./src/gui.o \
	./src/imgvu.o

# Additional CUDA includes if any
CU_INCS := \
    src/ten4_config.h \
    src/ten4_types.h \
	src/opt.h \
    src/util.h \
    src/vector.h \
	src/istream.h \
	src/ostream.h \
	src/aio.h \
	src/mmu.h \
	src/tlsf.h \
	src/tensor.h \
	src/vm.h \
	src/eforth.h \
	src/t4base.h \
	src/tenvm.h \
	src/netvm.h \
	src/model.h \
	src/ten4.h

# Cutlass includes
CL_INC  := ${CUTLASS_HOME}/include
CL_TOOL := ${CUTLASS_HOME}/tools

.PHONY: clean clean-src

# Each subdirectory must supply rules for building sources it contributes
src/%.o: src/%.cu ${CU_INCS}
	@echo 'Building source file: $<'
	@echo 'Invoking: NVCC Compiler'
	${CUDA_HOME}/bin/nvcc -ccbin g++ \
		${INC_PARMS} \
		-D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING \
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
