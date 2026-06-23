#
# tensorForth src/mu/module.mk
#
# Data-only fragment — no rules, no boilerplate.
# All compile logic lives in mk/module.mk.

MU_SRCS :=     \
	tlsf.cpp   \
	mpool.cpp  

MU_CUSRCS :=   \
	tensor.cu  \
	mmu.cu     \
	dataset.cu

MU_DIRS :=
MU_LIBS :=

$(eval $(call MODULE_RULE,MU,src/mu))
