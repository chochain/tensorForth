#
# tensorForth src/ld/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
LD_SRCS := \
	src/ld/mnist.cu \
	src/ld/loader.cu

LD_INCS := \
	src/ld/corpus.h \
	src/ld/mnist.h \
	src/ld/loader.h

LD_EXT := \
	src/ten4_types.h

LD_OBJS := $(LD_SRCS:%.cu=%.o)

OBJS += $(LD_OBJS)

.PHONY: src-ld clean-src-ld

src-ld: $(LD_OBJS)

# Each subdirectory must supply rules for building sources it contributes
src/ld/%.o: src/ld/%.cu $(LD_INCS) $(LD_EXT)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NVCC) $(NVCC_FLAGS) -o "$@" "$<" || $(NV_ERR)
	@echo '</Status></Source>'
	@echo ' '

clean-src-ld:
	-$(RM) $(LD_OBJS)
