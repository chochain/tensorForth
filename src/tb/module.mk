#
# tensorForth src/tb/module.mk
#
# Add inputs and outputs from these tool invocations to the build variables
TB_SRCS := \
	src/tb/summary.cu

TB_DIRS :=
TB_LIBS :=

$(eval $(call CUDA_MODULE,TB,src/tb))
