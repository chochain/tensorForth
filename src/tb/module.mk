#
# tensorForth src/tb/module.mk
#
# Add inputs and outputs from these tool invocations to the build variables
TB_SRCS := \
	summary.cpp

TB_CUSRCS :=

TB_DIRS :=
TB_LIBS :=

$(eval $(call MODULE_RULE,TB,src/tb))
