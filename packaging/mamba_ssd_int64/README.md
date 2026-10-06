# mamba-ssd-int64

Loader-only, pure-Python copy of mamba-ssm's Triton SSD kernels with the
int32-overflow fix for [state-spaces/mamba#1015](https://github.com/state-spaces/mamba/issues/1015).

- Source: this repository's `mamba_ssm/ops/triton/{ssd_bmm,ssd_chunk_state,
  ssd_state_passing,ssd_chunk_scan,layernorm_gated,k_activations,ssd_combined}.py`,
  shipped byte-identical. The branch is upstream `main` (e9594ce1) plus
  [state-spaces/mamba#1037](https://github.com/state-spaces/mamba/pull/1037) at
  head b97c7ca0 (stack #1038, #1034-#1037) and the layernorm_gated /
  k_activations hunks of [#1016](https://github.com/state-spaces/mamba/pull/1016)
  (head 732efd55).
- The files keep their absolute `mamba_ssm.ops.triton.*` imports, so importing
  one directly would bind the stock kernels of the installed mamba-ssm. Importing
  `mamba_ssd_int64.<file>` raises `ImportError`; execute the files from
  `mamba_ssd_int64.path(name)` with a `sys.modules` loader instead.
- Install: `pip install "git+https://github.com/dotomics/mamba@<sha>#subdirectory=packaging/mamba_ssd_int64"`.
  It needs mamba-ssm (helpers), torch and triton at runtime, and builds no CUDA.
- Retire once mamba-ssm releases the fix.
