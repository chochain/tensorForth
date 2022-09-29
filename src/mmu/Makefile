#
# tensorForth src/ldr/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
MMU_SRCS := \
	./src/mmu/util.cu \
	./src/mmu/tlsf.cu \
	./src/mmu/tensor.cu \
	./src/mmu/mmu.cu

MMU_OBJS := $(MMU_SRCS:%.cu=%.o)
MMU_INCS := \
	./src/mmu/util.h \
	./src/mmu/vector.h \
	./src/mmu/t4base.h \
	./src/mmu/tensor.h \
	./src/mmu/tlsf.h \
	./src/mmu/mmu.h

SRCS +=
INCS +=
OBJS += $(MMU_OBJS)

.PHONY: src-mmu clean-src-mmu

src-mmu: $(MMU_OBJS)

# Each subdirectory must supply rules for building sources it contributes
src/mmu/%.o: src/mmu/%.cu $(MMU_INCS)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NV_CC) -o "$@" "$<" --keep-dir $(APP_HOME)/src/ldr || $(NV_ERR)
	@echo '</Status></Source>'
	@echo ' '

clean-src-mmu:
	-$(RM) $(MMU_OBJS)
