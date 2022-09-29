#
# tensorForth src/io/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
IO_SRCS := \
	./src/io/aio.cu

IO_INCS := \
	src/io/istream.h \
	src/io/ostream.h \
	src/io/aio.h

IO_OBJS := $(IO_SRCS:%.cu=%.o)

OBJS += $(IO_OBJS)

.PHONY: src-io clean-src-io

src-io: $(IO_OBJS)

# Each subdirectory must supply rules for building sources it contributes
src/io/%.o: src/io/%.cu $(IO_INCS)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NV_CC) -o "$@" "$<" --keep-dir $(APP_HOME)/src/ldr || $(NV_ERR)
	@echo '</Status></Source>'
	@echo ' '

clean-src-io:
	-$(RM) $(IO_OBJS)
