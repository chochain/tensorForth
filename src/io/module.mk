#
# tensorForth src/io/module.mk
#
# Data-only fragment — no rules, no boilerplate.
# All compile logic lives in mk/module.mk.

IO_SRCS := \
	aio.cpp \
	aio_tensor.cpp \
	aio_model.cpp

IO_DIRS :=
IO_LIBS :=

$(eval $(call MODULE_RULE,IO,src/io))
