#
# tensorForth src/mu/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
MU_SRCS := \
	src/mu/tlsf.cu \
	src/mu/tensor.cu \
	src/mu/mmu.cu \
	src/mu/dataset.cu

MU_INCS := \
	src/ten4_config.h \
	src/ten4_types.h \
	src/debug.h \
	src/t4base.h \
	src/util.h \
	src/t4math.h \
	src/mu/vector.h \
	src/mu/tensor.h \
	src/mu/tlsf.h \
	src/mu/code.h \
	src/mu/dataset.h \
	src/mu/mmu.h

MU_OBJS := $(MU_SRCS:%.cu=%.o)

OBJS += $(MU_OBJS)

MU_EXT := \
	src/nn/model.h \
	src/ld/loader.h \
	src/ld/corpus.h

#MU_LIBS := -Isrc/nn
MU_LIBS :=

.PHONY: src-mu clean-src-mu

src-mu: $(MU_OBJS)

# Each subdirectory must supply rules for building sources it contributes
src/mu/%.o: src/mu/%.cu $(MU_INCS) $(MU_EXT)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NVCC) $(NVCC_FLAGS) $(MU_LIBS) -o "$@" "$<" || $(NV_ERR)
	@echo '</Status></Source>'
	@echo ' '

clean-src-mu:
	-$(RM) $(MU_OBJS)
