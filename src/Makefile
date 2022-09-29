#
# tensorForth src/Makefile
#
# Add inputs and outputs from these tool invocations to the build variables
T4_SRCS := \
	./src/ten4.cu

T4_INCS := \
    ./src/ten4_config.h \
    ./src/ten4_types.h \
	./src/opt.h \
	./src/ten4.h

T4_OBJS := $(T4_SRCS:%.cu=%.o)

OBJS += $(T4_OBJS)

# Every subdirectory with source files must be described here
SUBDIRS := nn vu ldr vm io mmu

T4_LIBS := $(SUBDIRS:%=-Isrc/%)

-include src/nn/Makefile
-include src/vu/Makefile
-include src/ldr/Makefile
-include src/vm/Makefile
-include src/io/Makefile
-include src/mmu/Makefile

.PHONY: src clean clean-src

src: $(SUBDIR:%=src-%)

src-t4: $(T4_OBJS) $(T4_INCS)

# Each subdirectory must supply rules for building sources it contributes
src/%.o: src/%.cu
	@echo '<Source><Action>Compile</Action><Filename>$<</Filename><Status>'
	-$(NV_CC) $(T4_LIBS) -o "$@" "$<" || $(NV_ERR)
	@echo '</Source>'
	@echo ' '

clean-src: $(SUBDIRS:%=clean-src-%)
	-$(RM) $(T4_OBJS)
