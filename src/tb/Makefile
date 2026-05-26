#
# tensorForth src/tb/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
TB_SRCS := \
	src/tb/summary.cu

TB_INCS := \
	src/ten4_config.h \
	src/ten4_types.h \
	src/tb/graph.h \
	src/tb/schema.h \
	src/tb/png.h \
	src/tb/crc32c.h \
	src/tb/encoder.h \
	src/tb/hparam.h \
	src/tb/types.h \
	src/tb/writer.h \
	src/tb/projector.h \
	src/tb/summary.h

TB_OBJS := $(TB_SRCS:%.cu=%.o)

OBJS += $(TB_OBJS)

TB_EXT := \
	src/mu/tensor.h \
	src/nn/model.h

TB_LIBS :=

.PHONY: src-tb clean-src-tb

src-tb: $(TB_OBJS)

# Each subdirectory must supply rules for building sources it contributes
src/tb/%.o: src/tb/%.cu $(TB_INCS) $(TB_EXT)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NVCC) $(NVCC_FLAGS) $(TB_LIBS) -o "$@" "$<" || $(NV_ERR)
	@echo '</Status></Source>'
	@echo ' '

clean-src-tb:
	-$(RM) $(TB_OBJS)
