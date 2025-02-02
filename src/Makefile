#
# tensorForth src/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
# CUDA nvcc compiler and linker flags
NVCC_FLAGS:= \
	-ccbin g++ \
	-D__CUDACC__ \
	-D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING \
	-Isrc $(GL_INCS:%=-I%) \
	-t=0 -c -std=c++17 -O3 \
	--device-c --extended-lambda --expt-relaxed-constexpr \
	--device-debug --debug --use_fast_math \
	-gencode arch=${CUDA_ARCH},code=${CUDA_CODE}

NV_CC:=${CUDA_HOME}/bin/nvcc $(NVCC_FLAGS)

T4_SRCS := \
	./src/sys.cu \
	./src/debug.cu \
	./src/ten4.cu

T4_INCS := \
	./src/ten4_config.h \
	./src/ten4_types.h \
	./src/opt.h \
	./src/sys.h \
	./src/debug.h \
	./src/t4base.h \
	./src/ten4.h

T4_OBJS := $(T4_SRCS:%.cu=%.o)

OBJS += $(T4_OBJS)

# Every subdirectory with source files must be described here
#SUBDIRS := nn vu ldr vm io mmu  # vu excluded after v4.x
#SUBDIRS := nn ldr vm io mmu
SUBDIRS := vm io mmu      # stream devel

T4_LIBS := $(SUBDIRS:%=-Isrc/%)

-include src/mmu/Makefile
-include src/io/Makefile
-include src/vm/Makefile
#-include src/ldr/Makefile
#-include src/vu/Makefile
#-include src/nn/Makefile

.PHONY: src clean clean-src

src: $(SUBDIR:%=src-%)

src-t4: $(T4_OBJS) $(T4_INCS)

# Each subdirectory must supply rules for building sources it contributes
src/%.o: src/%.cu
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	${NV_CC} $(T4_LIBS) -o "$@" "$<" || $(NV_ERR)
	@echo '</Source>'
	@echo ' '

clean-src: $(SUBDIRS:%=clean-src-%)
	-$(RM) $(T4_OBJS)
