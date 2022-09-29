
# tensorForth src/vu/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
VU_SRCS := \
	./src/vu/gui.cu \
	./src/vu/vu.cu \
	./src/vu/mnist_vu.cu

VU_OBJS := $(VU_SRCS:%.cu=%.o)
VU_INCS := \
	./src/ten4_types.h \
	./src/ldr/ndata.h \
	./src/vu/vu.h \
	./src/vu/mnist_vu.h

SRCS +=
INCS +=
OBJS += $(VU_OBJS)

.PHONY: src-vu clean-src-vu

src-vu: $(VU_OBJS)

# Each subdirectory must supply rules for building sources it contributes
src/vu/%.o: src/vu/%.cu $(VU_INCS)
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NV_CC) -o "$@" "$<" --keep-dir $(APP_HOME)/src/vu || echo 'FAILED'
	@echo '</Status></Source>'
	@echo ' '

clean-src-vu:
	-$(RM) $(VU_OBJS)
