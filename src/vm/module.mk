#
# tensorForth src/vm/module.mk
#
# Add inputs and outputs from these tool invocations to the build variables
VM_SRCS := \
	src/vm/vm.cu \
	src/vm/eforth.cu \
	src/vm/tenvm.cu \
	src/vm/netvm.cu

$(eval $(call MODULE_RULES,VM,src/vm))
