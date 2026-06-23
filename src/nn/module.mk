#
# tensorForth src/nn/module.mk
#
# Data-only fragment — no rules, no boilerplate.
# All compile logic lives in mk/module.mk.

NN_CUSRCS := \
	layer.cu \
	model.cu \
	network.cu

NN_DIRS :=
NN_LIBS :=

$(eval $(call MODULE_RULE,NN,src/nn))
