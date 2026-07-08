#
# tensorForth src/vm/module.mk
#
# Add inputs and outputs from these tool invocations to the build variables
VM_SRCS :=     \
	vm.cpp     \
	eforth.cpp \
	tenvm.cpp  \
	netvm.cpp

VM_CUSRCS :=

VM_DIRS :=
VM_LIBS :=

$(eval $(call MODULE_RULE,VM,src/vm))
