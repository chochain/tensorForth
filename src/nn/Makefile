#
# tensorForth src/nn/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
NN_SRCS := \
	src/nn/model.cu \
    src/nn/debug.cu \
	src/nn/forward.cu \
	src/nn/backprop.cu \
	src/nn/loss.cu

NN_INCS :=
NN_OBJS := $(NN_SRCS:%.cu=%.o)

OBJS += $(NN_OBJS)

NN_EXT := \
	src/mmu/t4base.h \
	src/mmu/tensor.h \
	src/mmu/mmu.h \
	src/mmu/model.h \
	src/mmu/dataset.h

NN_LIBS := -Isrc/mmu

.PHONY: src src-nn src-ds clean clean-src

src-nn: $(NN_OBJS) $(NN_INCS)

# Each subdirectory must supply rules for building sources it contributes
src/nn/%.o: src/nn/%.cu $(NN_INCS) $(NN_EXT)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NV_CC) $(NN_LIBS) -o "$@" "$<" || $(NV_ERR)
	@echo '</Source>'
	@echo ' '

clean-src-nn:
	-$(RM) $(NN_OBJS)
