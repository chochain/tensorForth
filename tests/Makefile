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
	t_mnist

TSTS := \
	t_bmpvu	\
	t_imgvu \
	t_mnist

TOBJS := \
	src/mmu/util.o \
	src/mmu/tlsf.o \
	src/mmu/mmu.o \
	src/mmu/tensor.o \
	src/vu/vu.o \
	src/vu/gui.o \
	src/vu/mnist_vu.o \
	src/ldr/mnist.o \
	src/ldr/loader.o

TST_LIBS:= -Isrc/ldr -Isrc/vu

.PHONY: test clean-test

test: $(TSTS:%.cu=%.o)

# Each subdirectory must supply rules for building sources it contributes
tests/%.o: tests/%.cu $(T4_INCS) $(DS_INCS)
	@echo '<Test><Action>Compile></Action><Filename>$<</Filename><Status>'
	-$(NV_CC) $(TST_LIBS) --keep-dir ${APP_HOME}/tests -o "$@" "$<"
	@echo '</Status></Test>'
	@echo ' '

$(TSTS): $(TSTS:%=tests/%.o)
	@echo '<Test><Action>Link</Action><Filename>$@</Filename><Status>'
	-$(NV_LNK) -lcusolver -o "./tests/$(@:.o=)" "./tests/$@.o" $(TOBJS)
	@echo '</Status></Test>'
	@echo ' '

clean-tst:
	-$(RM) $(TSTS:%=tests/%.o)
