#
# tensorForth tests/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
TSTS := \
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

TOBJS := \
	./src/util.o \
	./src/tlsf.o \
	./src/mmu.o \
	./src/tensor.o \
	./src/vu.o \
	./src/gui.o

.PHONY: test clean-test

test: $(TSTS)

# Each subdirectory must supply rules for building sources it contributes
tests/%.o: tests/%.cu $(TOBJS) $(T4_INCS) $(DS_INCS)
	@echo '<Test><Action>Compile></Action><Filename>$<</Filename>'
	$(NV_CC) -o "$@" "$<" --keep-dir ${APP_HOME}/tests
	@echo '<Status>OK</Status></Test>'
	@echo ' '

$(TSTS): $(TSTS:%=tests/%.o) 
	@echo '<Test><Action>Link</Action><Filename>$@</Filename>'
	$(NV_LNK) -lcusolver -o "./tests/$(@:.o=)" "./tests/$@.o" $(TOBJS)
	@echo '<Status>OK</Status></Test>'
	@echo ' '

clean-tst:
	-$(RM) $(TSTS:%=./tests/%.o)
