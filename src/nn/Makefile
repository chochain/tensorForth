#
# tensorForth src/nn/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
NN_SRCS := \
	src/nn/model.cu \
	src/nn/forward.cu \
	src/nn/backprop.cu

NN_INCS := \
	src/nn/model.h \
	src/nn/dataset.h

NN_OBJS := $(NN_SRCS:%.cu=%.o)

OBJS += $(NN_OBJS)

NN_EXT := \
	src/mmu/tensor.h \
	src/mmu/mmu.h

NN_LIBS := -Isrc/mmu

.PHONY: src src-nn src-ds clean clean-src

src-nn: $(NN_OBJS) $(NN_INCS)

# Each subdirectory must supply rules for building sources it contributes
src/nn/%.o: src/nn/%.cu
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NV_CC) $(NN_LIBS) -o "$@" "$<" --keep-dir $(APP_HOME)/src || $(NV_ERR)
	@echo '</Source>'
	@echo ' '

clean-src-nn:
	-$(RM) $(NN_OBJS)
