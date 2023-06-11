#
# tensorForth root Makefile
#
-include ${HOME}/makefile.init

RM := rm

APP_NAME  := ten4
APP_HOME  := ${HOME}/devel/forth/$(APP_NAME)
APP_TGT   := $(APP_HOME)/tests/$(APP_NAME)

CUDA_LIB  := ${CUDA_HOME}/targets/x86_64-linux/lib
CUDA_ARCH := compute_75
CUDA_CODE := sm_75
CUDA_ARCH1:= compute_52
CUDA_CODE1:= sm_52
CUDA_FLAGS:= -Xnvlink --suppress-stack-size-warning --cudart=static

# All of the sources participating in the build are defined here
SRCS := src/ten4.cu
OBJS := $(SRCS:%.cu=%.o)

# Cutlass library in ${CUTLASS_HOME}/build/tools/library
CL_LIB  := -L${CUTLASS_HOME}/build/tools/library -lcutlass
CL_TOOL := ${CUTLASS_HOME}/tools
CL_INCS := \
	${CUTLASS_HOME}/include \
	${CUTLASS_HOME}/tools/library/include \
	${CUTLASS_HOME}/tools/util/include

# GL libraries
GL_LIB  := -lGL -lGLU -lglut -lX11
GL_INCS := \
    /u01/src/stb \
	${CUDA_HOME}/cuda-samples/Common \
	${CUDA_HOME}/cuda-samples/Samples/2_Concepts_and_Techniques/imageDenoising

# CUDA nvcc compiler and linker flags
NV_CC   := \
	${CUDA_HOME}/bin/nvcc -ccbin g++ \
	-D__CUDACC__ \
	-D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING \
	-Isrc $(CL_INCS:%=-I%) $(GL_INCS:%=-I%) \
	-t=0 -c -std=c++14 -O3 \
	--device-c --extended-lambda --expt-relaxed-constexpr \
	--device-debug --debug --use_fast_math \
	-gencode arch=${CUDA_ARCH},code=${CUDA_CODE}

NV_LNK := \
	${CUDA_HOME}/bin/nvcc -ccbin g++ \
	$(CUDA_FLAGS) \
	-L$(CUDA_LIB) \
	$(GL_LIB) \
	-gencode arch=${CUDA_ARCH},code=${CUDA_CODE}

NV_ERR := echo "NVCC_FAILED"

# Add inputs and outputs from these tool invocations to the build variables
-include src/Makefile
-include tests/Makefile

# Extra pre-compiled object and libraries
USER_OBJS :=
USER_LIBS :=

OPTIONAL_TOOL_DEPS := \
	$(wildcard ${HOME}/makefile.defs) \
	$(wildcard ${HOME}/makefile.init) \
	$(wildcard ${HOME}/makefile.targets)

.PHONY: all tests clean

# All Target
all: src $(APP_NAME)

tests: test

# Tool invocations
$(APP_NAME): $(OBJS) $(USER_OBJS) $(OPTIONAL_TOOL_DEPS)
	@echo '<App><Action>Link</Action><Filename>$<</Filename><Status>'
	-$(NV_LNK) -o "$(APP_TGT)" $^ || echo $(NV_ERR)
	@echo '</Status></App>'
	@echo ' '

clean: clean-src clean-tst
	-$(RM) $(APP_TGT)
	@echo ' '

# other targets
-include ../makefile.targets
