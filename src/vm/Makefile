#
# tensorForth src/vm/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
VM_SRCS := \
	src/vm/vm.cu \
	src/vm/eforth.cu \
	src/vm/tenvm.cu \
	src/vm/netvm.cu

VM_INCS := \
	src/vm/vm.h \
	src/vm/param.h \
	src/vm/eforth.h \
	src/vm/tenvm.h \
	src/vm/netvm.h

VM_OBJS := $(VM_SRCS:%.cu=%.o)

OBJS += $(VM_OBJS)

VM_EXT := \
	src/ten4_config.h \
	src/ten4_types.h \
	src/io/aio.h \
	src/mmu/tensor.h \
	src/nn/dataset.h \
	src/nn/model.h

#VM_LIBS:= -Isrc/io -Isrc/mmu -Isrc/nn
VM_LIBS:=

.PHONY: src-vm clean-src-vm

src-vm: $(VM_OBJS)

# Each subdirectory must supply rules for building sources it contributes
src/vm/%.o: src/vm/%.cu $(VM_INCS) $(VM_EXT)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NVCC) $(NVCC_FLAGS) $(VM_LIBS) -o "$@" "$<" || $(NV_ERR)
	@echo '</Status></Source>'
	@echo ' '

clean-src-vm:
	-$(RM) $(VM_OBJS)
