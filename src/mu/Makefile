#
# tensorForth src/mu/module.mk
#
# Data-only fragment — no rules, no boilerplate.
# All compile logic lives in mk/module.mk.

MU_SRCS := \
	src/mu/tlsf.cu \
	src/mu/mpool.cu \
	src/mu/tensor.cu \
	src/mu/mmu.cu \
	src/mu/dataset.cu

MU_DIRS :=
MU_LIBS :=

$(eval $(call CUDA_MODULE,MU,src/mu))
