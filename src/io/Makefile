#
# tensorForth src/io/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
IO_SRCS := \
	src/io/aio.cu \
    src/io/aio_tensor.cu \
    src/io/aio_model.cu

IO_INCS := \
	src/io/istream.h \
	src/io/ostream.h \
	src/io/aio.h

IO_OBJS := $(IO_SRCS:%.cu=%.o)

# extra include from other directries
IO_EXT:= \
	src/ten4_config.h \
	src/ten4_types.h \
    src/tbase.h \
	src/util.h \
	src/mmu/mmu.h \
	src/mmu/tensor.h \
	src/nn/dataset.h \
	src/nn/model.h \
	src/ldr/corpus.h \
    src/ldr/loader.h

#IO_LIBS := -Isrc/mmu -Isrc/ldr -I/src/nn
IO_LIBS :=

OBJS += $(IO_OBJS)

.PHONY: src-io clean-src-io

src-io: $(IO_OBJS)

# Each subdirectory must supply rules for building sources it contributes
src/io/%.o: src/io/%.cu $(IO_INCS) $(IO_EXT)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NV_CC) $(IO_LIBS) -o "$@" "$<" || $(NV_ERR)
	@echo '</Status></Source>'
	@echo ' '

clean-src-io:
	-$(RM) $(IO_OBJS)
