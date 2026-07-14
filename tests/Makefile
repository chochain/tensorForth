#
# tensorForth tests/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
NVCC       := $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS := -ccbin g++ \
	          -D__CUDACC__ \
	          -Isrc $(GL_INCS:%=-I%) \
	          -t 0 -c -std=c++17 -O2 -g -lineinfo \
	          --device-c --expt-extended-lambda \
	          -Xptxas -v \
	          -gencode arch=$(CUDA_ARCH),code=$(CUDA_CODE)

TSTS0 := \
	t_tensor \
	t_tlsf \
	t_mmu_tensor \
	t_inverse \
	t_solver \
	t_gl \
	t_cg \
	t_bmpvu	\
	t_imgvu \
	t_mnist \
	t_ganvu \
	t_event

TSTS := \
	t_bmpvu

#t_imgvu \
#	t_mnist \
#	t_ganvu \
#	t_event

VU_OBJS :=               \
	../src/vu/vu.o       \
	../src/vu/gui.o      \
	../src/vu/interop.o  \
	../src/vu/mnist_vu.o

TOBJS:= $(VU_OBJS) $(LDR_OBJS)
TST_LIBS:= $(VU_LIBS) $(LDR_LIBS)

.PHONY: test clean-test

test: $(TSTS:%=%.o) $(TSTS)
	@echo "$<"

# Each subdirectory must supply rules for building sources it contributes
$(TSTS:%=%.o): $(TSTS:%=%.cu)
	@echo '<Test><Action>Compile></Action><Filename>$<</Filename><Status>'
	-$(NVCC) $(NVCC_FLAGS) $(TST_LIBS) --keep-dir ${APP_HOME}/tests \
	-o "$@" "$(@:.o=.cu)" || $(NV_ERR)
	@echo '</Status></Test>'
	@echo ' '

$(TSTS): $(TSTS:%=%.o) $(TOBJS)
	@echo '<Test><Action>Link</Action><Filename>$@</Filename><Status>'
	$(NVCC) $(NVLINK_FLAGS) -o $(@:.o=) $@.o $(TOBJS)
	@echo '</Status></Test>'
	@echo ' '

clean-test:
	-$(RM) $(TSTS:%=tests/%.o)
