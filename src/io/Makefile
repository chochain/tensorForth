#
# tensorForth src/io/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
IO_SRCS := \
	src/io/aio.cpp \
	src/io/aio_tensor.cpp \
	src/io/aio_model.cpp

IO_INCS := \
	src/io/istream.h \
	src/io/ostream.h \
	src/io/aio.h

IO_OBJS := $(IO_SRCS:%.cpp=%.o)

# extra include from other directries
IO_EXT:= \
	src/ten4_config.h \
	src/ten4_types.h \
	src/tbase.h \
	src/util.h \
	src/mu/tensor.h \
	src/nn/model.h

#IO_LIBS := -Isrc/mu -Isrc/nn -Isrc/tb
IO_LIBS :=

OBJS += $(IO_OBJS)

.PHONY: src-io clean-src-io

src-io: $(IO_OBJS)

# Each subdirectory must supply rules for building sources it contributes
src/io/%.o: src/io/%.cpp $(IO_INCS) $(IO_EXT)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(CC) $(CC_FLAGS) $(IO_LIBS) -o "$@" "$<" || $(CC_ERR)
	@echo '</Status></Source>'
	@echo ' '

clean-src-io:
	-$(RM) $(IO_OBJS)
