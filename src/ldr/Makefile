#
# tensorForth src/ldr/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
LDR_SRCS := \
	src/ldr/mnist.cu \
	src/ldr/loader.cu

LDR_INCS := \
	src/ten4_types.h \
	src/ldr/corpus.h \
	src/ldr/mnist.h \
	src/ldr/loader.h

LDR_OBJS := $(LDR_SRCS:%.cu=%.o)

OBJS += $(LDR_OBJS)

.PHONY: src-ldr clean-src-ldr

src-ldr: $(LDR_OBJS)

# Each subdirectory must supply rules for building sources it contributes
src/ldr/%.o: src/ldr/%.cu $(LDR_INCS)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NV_CC) -o "$@" "$<" || $(NV_ERR)
	@echo '</Status></Source>'
	@echo ' '

clean-src-ldr:
	-$(RM) $(LDR_OBJS)
