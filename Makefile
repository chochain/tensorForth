#
# tensorForth root Makefile
#
APP_NAME  := ten4
APP_HOME  := $(HOME)/devel/forth/$(APP_NAME)
APP_TGT   := $(APP_HOME)/tests/$(APP_NAME)

# GeForce GTX 1660, 1650 (Turing)
CUDA_ARCH := compute_75
CUDA_CODE := sm_75
# GeFroce GT 1030 (Pascal)
#CUDA_ARCH := compute_61
#CUDA_CODE := sm_61
# Jetson Nano (Maxwell)
#CUDA_ARCH:= compute_52
#CUDA_CODE:= sm_52
# older card (Kepler)
#CUDA_ARCH:= compute_35
#CUDA_CODE:= sm_35

# ── Toolchain ─────────────────────────────────────────────────────────────────
NVCC       := $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS := -ccbin g++ \
	          -D__CUDACC__ \
	          -Isrc $(GL_INCS:%=-I%) \
	          -t 0 -c -std=c++17 -O0 -g -lineinfo \
	          --device-c --expt-extended-lambda \
	          -Xptxas -v \
	          -gencode arch=$(CUDA_ARCH),code=$(CUDA_CODE)
NV_ERR     := echo "NVCC_FAILED"

CC         := g++
CC_FLAGS   := -std=c++17 -c -g -O3 -Wall \
	          -Isrc $(GL_INCS:%=-I%) \
	          -fomit-frame-pointer -fno-stack-check -fno-stack-protector \
	          -march=native -ffast-math -funroll-loops
CC_ERR     := echo "CC_FAILED"

RM         := rm -f

# ── Output binary ─────────────────────────────────────────────────────────────
CUDA_LIB   := -L$(CUDA_HOME)/targets/x86_64-linux/lib
CL_LIBS    :=
#CL_LIBS    := -L$(CUTLASS_HOME)/build/tools/library -lcutlass
GL_LIBS    :=
#GL_LIBS    := -lGL -lGLU -lglut -lX11 -lcudadevrt

#LINK_FLAGS := -lcudart -lcublas            # adjust per your CUDA libs
LINK_FLAGS := -ccbin g++ \
              -Xnvlink --suppress-stack-size-warning \
              $(CUDA_LIB) $(CL_LIBS) $(GL_LIBS) \
              -gencode arch=$(CUDA_ARCH),code=$(CUDA_CODE)

# IMPORTANT: must be declared before any module fragment is included,
# so that each module's $(eval $(call ..)) can safely += into it.
OBJS       :=

# ── Module registry ───────────────────────────────────────────────────────────
# To add a new module: drop a src/<mod>/module.mk, append the name here.
MODULES    := mu io vm ld nn tb

# ── Load shared rule template (must come before module fragments) ──────────────
include module_rules.mk

# ── Load each module's data fragment (populates OBJS via +=) ──────────────────
#include src/module.mk
include $(MODULES:%=src/%/module.mk)

# ── Top-level targets ─────────────────────────────────────────────────────────
.PHONY: all clean

# Compile all modules, then link into the final binary
all: $(APP_NAME)

show:
	@echo $(MODULES:%=src/%/module.mk)

# Link step: nvcc drives the link so CUDA device code is resolved correctly.
# Depends on all module .o files accumulated in OBJS.
$(APP_NAME): $(OBJS)
	@echo '<App><Action>Link</Action><Filename>$<</Filename><Status>'
	$(NVCC) $(LINK_FLAGS) $(OBJS) -o $(APP_TGT) $^ || $(NV_ERR)
	@echo '</Status></App>'
	@echo 'Built: $@'

clean: $(MODULES:%=clean-src-%)
	-$(RM) $(APP_NAME)

# ── Auto-generated header dependencies ────────────────────────────────────────
# Compiler writes .d files alongside .o files via -MMD -MP.
# Include them so header changes trigger the right recompiles.
# The leading '-' silences "file not found" on a clean build.
DEPS :=
-include $(DEPS)
