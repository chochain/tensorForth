#
# tensorForth src/ld/module.mk
#
# Data-only fragment — no rules, no boilerplate.
# All compile logic lives in mk/module.mk.

LD_SRCS := \
	loader.cpp \
	mnist.cpp \
	cifar10.cpp

LD_DIRS :=
LD_LIBS :=

$(eval $(call MODULE_RULE,LD,src/ld))
