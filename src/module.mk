#
# tensorForth src/module.mk
#
# Add inputs and outputs from these tool invocations to the build variables
#
T4_CUSRCS :=  \
	sys.cu    \
	debug.cu  \
	util.cu   \
	t4math.cu \
	ten4.cu

T4_DIRS :=
T4_LIBS :=

$(eval $(call MODULE_RULE,T4,src))
