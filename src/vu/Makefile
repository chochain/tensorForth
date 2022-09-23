#
# tensorForth src/vu/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
VU_OBJS := \
	./src/vu/gui.o \
	./src/vu/vu.o \
	./src/vu/mnist_vu.o

VU_INCS := \
	./src/ten4_types.h \
	./src/dataset.h \
	./src/vu/vu.h

SRCS +=
INCS +=
OBJS += $(VU_OBJS)

.PHONY: src-vu clean-src-vu

src-vu: $(VU_OBJS)

# Each subdirectory must supply rules for building sources it contributes
src/vu/%.o: src/vu/%.cu $(VU_INCS)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename>'
	$(NV_CC) -o "$@" "$<" --keep-dir $(APP_HOME)/src/vu
	@echo '<Status>OK</Status></Source>'
	@echo ' '

clean-src-vu:
	-$(RM) $(VU_OBJS)
