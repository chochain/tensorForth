#
# tensorForth src/module.mk
#
# Add inputs and outputs from these tool invocations to the build variables
#
T4_SRCS := \
	./src/sys.cu \
	./src/debug.cu \
	./src/util.cu \
	./src/t4math.cu \
	./src/ten4.cu

T4_DIRS := src src/io src/mu src/vm src/nn src/tb
T4_LIBS :=

$(eval $(call MODULE_RULES,T4,src))
