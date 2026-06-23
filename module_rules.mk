#
# tensorForth mk/module.mk
#
# Shared compile rule template for all sub-modules (.cu and .cpp).
#
# Usage in each src/<mod>/module.mk:
#   1. Define <MOD>_SRCS, <MOD>_DIRS
#   2. Call: $(eval $(call CUDA_MODULE,MOD,src/mod))
#          or $(eval $(call CPP_MODULE,MOD,src/mod))
#
# Header dependencies are tracked automatically via compiler-generated .d files
# (-MMD -MP flags). No need to list headers manually in the pattern rule.

# Helper: uppercase -> lowercase (single line, GNU make 3.81+)
lc = $(subst A,a,$(subst B,b,$(subst C,c,$(subst D,d,$(subst E,e,$(subst F,f,$(subst G,g,$(subst H,h,$(subst I,i,$(subst J,j,$(subst K,k,$(subst L,l,$(subst M,m,$(subst N,n,$(subst O,o,$(subst P,p,$(subst Q,q,$(subst R,r,$(subst S,s,$(subst T,t,$(subst U,u,$(subst V,v,$(subst W,w,$(subst X,x,$(subst Y,y,$(subst Z,z,$1))))))))))))))))))))))))))

# ── Shared scaffolding ────────────────────────────────────────────────────────
# $(1) MODULE prefix,  $(2) source directory
define _MODULE_BASE
$(1)_OBJS   := $$($(1)_SRCS:%.cpp=$(2)/%.o)
$(1)_DEPS   := $$($(1)_SRCS:%.cpp=$(2)/%.d)
$(1)_CUOBJS := $$($(1)_CUSRCS:%.cu=$(2)/%.o)
$(1)_CUDEPS := $$($(1)_CUSRCS:%.cu=$(2)/%.d)
$(1)_INCS   := $$($(1)_DIRS:%=-I%)
OBJS        += $$($(1)_OBJS) $$($(1)_CUOBJS)
DEPS        += $$($(1)_DEPS) $$($(1)_CUDEPS)

.PHONY: src-$(call lc,$(1)) clean-src-$(call lc,$(1))

src-$(call lc,$(1)): $$($(1)_OBJS) $$($(1)_CUOBJS)

clean-$(call lc,$(1)):
	-$$(RM) $$($(1)_OBJS) $$($(1)_DEPS) $$($(1)_CUOBJS) $$($(1)_CUDEPS)
endef

# ── CUDA module: .cu -> .o via nvcc ──────────────────────────────────────────
# -MMD -MP: auto-generate .d dependency files so header changes trigger recompile
define MODULE_RULE
$(call _MODULE_BASE,$(1),$(2))

$(2)/%.o: $(2)/%.cpp
	@echo '<Source><Action>Compile</Action><Filename>$$<</Filename><Status>'
	-$$(CC) $$(CC_FLAGS) $$($(1)_INCS) -MMD -MP $$($(1)_LIBS) -o "$$@" "$$<" || $$(CC_ERR)
	@echo '</Status></Source>'
	@echo ' '

$(2)/%.o: $(2)/%.cu
	@echo '<Source><Action>Compile</Action><Filename>$$<</Filename><Status>'
	-$$(NVCC) $$(NVCC_FLAGS) $$($(1)_INCS) -MMD -MP $$($(1)_LIBS) -o "$$@" "$$<" || $$(NV_ERR)
	@echo '</Status></Source>'
	@echo ' '
endef
