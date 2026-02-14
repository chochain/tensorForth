#
# tensorForth root Makefile
#
-include ${HOME}/makefile.init

RM := rm

APP_NAME  := ten4
APP_HOME  := ${HOME}/devel/forth/$(APP_NAME)
APP_TGT   := $(APP_HOME)/tests/$(APP_NAME)

CUDA_LIB  := ${CUDA_HOME}/targets/x86_64-linux/lib
# GeForce GTX 1660, 1650 (Turing)
#CUDA_ARCH := compute_75
#CUDA_CODE := sm_75
# GeFroce GT 1030 (Pascal)
CUDA_ARCH := compute_61
CUDA_CODE := sm_61
# Jetson Nano (Maxwell)
#CUDA_ARCH:= compute_52
#CUDA_CODE:= sm_52
# older card (Kepler)
#CUDA_ARCH:= compute_35
#CUDA_CODE:= sm_35

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

# GL libraries (deprecated v4.x, i.e. separation of View from M and C)
#GL_LIB  :=
#GL_INCS :=
GL_LIB  := -lGL -lGLU -lglut -lX11
GL_INCS := \
	/u01/src/stb \
	${CUDA_HOME}/cuda-samples/Common \
	${CUDA_HOME}/cuda-samples/Samples/2_Concepts_and_Techniques/imageDenoising

NVCC:=${CUDA_HOME}/bin/nvcc
NVCC_FLAGS:= \
	-ccbin g++ \
	-D__CUDACC__ \
	-Isrc $(GL_INCS:%=-I%) \
	-t=0 -c -std=c++17 -O2 \
	--device-c --extended-lambda \
	--expt-relaxed-constexpr \
	-gencode arch=${CUDA_ARCH},code=${CUDA_CODE}

#	--device-debug --debug --use_fast_math \

NVLINK_FLAGS:= \
	-ccbin g++ \
	-Xnvlink --suppress-stack-size-warning --cudart=static \
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
	$(NVCC) $(NVLINK_FLAGS) -o $(APP_TGT) $^ || $(NV_ERR)
	@echo '</Status></App>'
	@echo ' '

clean: clean-src clean-test
	-$(RM) $(APP_TGT)
	@echo ' '

# other targets
-include ../makefile.targets
