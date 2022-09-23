#
# tensorForth src/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
SRCS +=

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
	./src/vu.o \
    ./src/gui.o \
	./src/ten4.o

# Additional CUDA includes if any
T4_INCS := \
    src/ten4_config.h \
    src/ten4_types.h \
	src/opt.h \
    src/util.h \
	src/t4base.h \
    src/vector.h \
	src/istream.h \
	src/ostream.h \
	src/tlsf.h \
	src/tensor.h \
	src/mmu.h \
	src/aio.h

VM_INCS := \
	src/vm.h \
	src/eforth.h \
	src/tenvm.h \
	src/model.h \
	src/netvm.h \
	src/ten4.h

DS_INCS := \
	src/dataset.h \
	src/vu.h \
	src/gui.h

-include src/ldr/Makefile
-include src/vu/Makefile

.PHONY: src clean clean-src

src: $(OBJS) src-ldr src-vu

# Each subdirectory must supply rules for building sources it contributes
src/%.o: src/%.cu $(T4_INCS) $(VM_INCS) $(DS_INCS)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename>'
	$(NV_CC) -o "$@" "$<" --keep-dir $(APP_HOME)/src
	@echo '<Status>OK</Status></Source>'
	@echo ' '

clean-src: clean-src-ldr clean-src-vu
	-$(RM) $(OBJS)
