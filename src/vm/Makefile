#
# tensorForth src/vm/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
VM_SRCS := \
	./src/vm/vm.cu \
	./src/vm/eforth.cu \
	./src/vm/tenvm.cu \
	./src/vm/netvm.cu

VM_OBJS := $(VM_SRCS:%.cu=%.o)
VM_INCS := \
	./src/vm/vm.h \
	./src/vm/eforth.h \
	./src/vm/tenvm.h \
	./src/vm/netvm.h

SRCS +=
INCS +=
OBJS += $(VM_OBJS)

.PHONY: src-vm clean-src-vm

src-vm: $(VM_OBJS)

# Each subdirectory must supply rules for building sources it contributes
src/vm/%.o: src/vm/%.cu $(VM_INCS)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NV_CC) -o "$@" "$<" --keep-dir $(APP_HOME)/src/ldr || $(NV_ERR)
	@echo '</Status></Source>'
	@echo ' '

clean-src-vm:
	-$(RM) $(VM_OBJS)
