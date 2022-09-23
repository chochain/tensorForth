#
# tensorForth src/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
T4_SRCS := \
	./src/util.cu \
	./src/aio.cu \
	./src/mmu.cu \
	./src/tlsf.cu \
	./src/tensor.cu
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

VM_SRCS := \
	./src/vm.cu \
	./src/eforth.cu \
	./src/tenvm.cu \
	./src/model.cu \
	./src/model_fwd.cu \
	./src/model_back.cu \
	./src/netvm.cu
VM_INCS := \
	src/vm.h \
	src/eforth.h \
	src/tenvm.h \
	src/model.h \
	src/netvm.h \
	src/dataset.h

TS_OBJS := $(T4_SRCS:%.cu=%.o)
VM_OBJS := $(VM_SRCS:%.cu=%.o)

OBJS += $(TS_OBJS) $(VM_OBJS)

-include src/ldr/Makefile
-include src/vu/Makefile

.PHONY: src src-t4 src-vm src-ds clean clean-src

src: src-ldr src-vu

src-t4: $(T4_OBJS) $(T4_INCS)

src-vm: $(VM_OBJS) src-t4 $(VM_INCS)

# Each subdirectory must supply rules for building sources it contributes
src/%.o: src/%.cu
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename>'
	-$(NV_CC) -o "$@" "$<" --keep-dir $(APP_HOME)/src
	@echo '<Status>OK</Status></Source>'
	@echo ' '

clean-src: clean-src-ldr clean-src-vu
	-$(RM) $(OBJS)
