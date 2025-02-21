#
# tensorForth src/ldr/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
MMU_SRCS := \
	src/mmu/tlsf.cu \
	src/mmu/tensor.cu \
	src/mmu/mmu.cu

MMU_INCS := \
	src/ten4_config.h \
	src/ten4_types.h \
	src/debug.h \
	src/t4base.h \
	src/mmu/vector.h \
	src/mmu/tensor.h \
	src/mmu/tlsf.h \
	src/mmu/code.h \
	src/mmu/mmu.h \
	src/mmu/model.h \
	src/mmu/dataset.h

MMU_OBJS := $(MMU_SRCS:%.cu=%.o)

OBJS += $(MMU_OBJS)

MMU_EXT := \
	src/ldr/loader.h

MMU_LIBS :=

.PHONY: src-mmu clean-src-mmu

src-mmu: $(MMU_OBJS)

# Each subdirectory must supply rules for building sources it contributes
src/mmu/%.o: src/mmu/%.cu $(MMU_INCS) $(MMU_EXT)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NV_CC) $(MMU_LIBS) -o "$@" "$<" || $(NV_ERR)
	@echo '</Status></Source>'
	@echo ' '

clean-src-mmu:
	-$(RM) $(MMU_OBJS)
