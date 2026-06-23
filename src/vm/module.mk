#
# tensorForth src/vm/module.mk
#
# Add inputs and outputs from these tool invocations to the build variables
VM_CUSRCS := \
	vm.cu \
	eforth.cu \
	tenvm.cu \
	netvm.cu

VM_DIRS :=
VM_LIBS :=

$(eval $(call MODULE_RULE,VM,src/vm))
