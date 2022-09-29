#
# tensorForth src/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
T4_SRCS := \
	./src/util.cu \
	./src/tlsf.cu \
	./src/tensor.cu \
	./src/mmu.cu 
T4_INCS := \
    src/ten4_config.h \
    src/ten4_types.h \
	src/opt.h \
    src/util.h \
	src/t4base.h \
    src/vector.h \
	src/tlsf.h \
	src/tensor.h \
	src/mmu.h

VM_SRCS := \
	./src/vm.cu \
	./src/eforth.cu \
	./src/tenvm.cu
VM_INCS := \
	src/vm.h \
	src/eforth.h \
	src/tenvm.h

T4_OBJS := $(T4_SRCS:%.cu=%.o)
VM_OBJS := $(VM_SRCS:%.cu=%.o)

OBJS += $(T4_OBJS) $(VM_OBJS)

-include src/io/Makefile
-include src/ldr/Makefile
-include src/vu/Makefile
-include src/nn/Makefile

.PHONY: src src-t4 src-vm src-ds clean clean-src

src: src-t4 src-vm src-ldr src-vu src-nn

src-t4: $(T4_OBJS) $(T4_INCS)

src-vm: $(VM_OBJS) src-t4 $(VM_INCS)

# Each subdirectory must supply rules for building sources it contributes
src/%.o: src/%.cu
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NV_CC) -o "$@" "$<" --keep-dir $(APP_HOME)/src || $(NV_ERR)
	@echo '</Source>'
	@echo ' '

clean-src: clean-src-io clean-src-ldr clean-src-vu clean-src-nn
	-$(RM) $(OBJS)
