
# tensorForth src/vu/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
VU_CUSRCS :=    \
	gui.cu      \
	vu.cu       \
	mnist_vu.cu

VU_DIRS :=
VU_LIBS :=

$(eval $(call MODULE_RULE,VU,src/vu))
