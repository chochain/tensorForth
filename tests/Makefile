#
# tensorForth tests/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
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
	t_bmpvu \
	t_imgvu \
	t_mnist \
	t_ganvu \
	t_event

TOBJS0 := \
	src/mmu/util.o \
	src/mmu/tlsf.o \
	src/mmu/mmu.o \
	src/mmu/tensor.o \
	src/vu/vu.o \
	src/vu/gui.o \
	src/vu/mnist_vu.o \
	src/ldr/mnist.o \
	src/ldr/loader.o

-include src/vu/Makefile

TOBJS:= $(VU_OBJS) $(LDR_OBJS)
TST_LIBS:= $(VU_LIBS) $(LDR_LIBS)

.PHONY: test clean-test

test: src-vu $(TSTS:%=tests/%.o) $(TSTS)
	@echo "$<"

# Each subdirectory must supply rules for building sources it contributes
$(TSTS:%=tests/%.o): $(TSTS:%=tests/%.cu)
	@echo '<Test><Action>Compile></Action><Filename>$<</Filename><Status>'
	-$(NVCC) $(NVCC_FLAGS) $(TST_LIBS) --keep-dir ${APP_HOME}/tests \
	-o "$@" "$(@:.o=.cu)" || $(NV_ERR)
	@echo '</Status></Test>'
	@echo ' '

$(TSTS): $(TSTS:%=tests/%.o) $(TOBJS)
	@echo '<Test><Action>Link</Action><Filename>$@</Filename><Status>'
	$(NVCC) $(NVLINK_FLAGS) -o "./tests/$(@:.o=)" "./tests/$@.o" $(TOBJS)
	@echo '</Status></Test>'
	@echo ' '

clean-test:
	-$(RM) $(TSTS:%=tests/%.o)
