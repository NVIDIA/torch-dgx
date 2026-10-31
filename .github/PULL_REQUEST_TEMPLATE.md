## Summary

<!-- Explain what changed, why it is needed, and the user-visible effect. -->

## Related issue or proposal

<!-- Link any related issues or proposals, or explain why none applies. -->

## Validation

<!--
Describe any tests added or updated, then list the exact validation commands and results.
For code changes, build with `uv pip install --no-build-isolation -e . --link-mode=copy`
and run `pytest -v -s tests/` in an environment with CUDA devices available.
Include GPU models and count, Python, PyTorch, and CUDA versions where relevant.
Use CPU PyTorch references and follow the Test Comparison Policy in CONTRIBUTING.md.
Explain any test or check that was not applicable or could not be run and why.
-->

## Additional context

<!-- Optional: Add anything else reviewers need, such as screenshots, documentation or migration guidance, release-note updates, and relevant compatibility, performance, security, or hardware considerations. Do not include sensitive vulnerability details; follow SECURITY.md instead. -->

## Checklist

- [ ] I reviewed the [contribution guidelines](https://github.com/NVIDIA/torch-dgx/blob/master/CONTRIBUTING.md).
- [ ] I have emailed the completed and signed NVIDIA [ICLA](https://github.com/NVIDIA/torch-dgx/blob/master/ICLA.pdf) or [CCLA](https://github.com/NVIDIA/torch-dgx/blob/master/CCLA.pdf) to [torch-dgx-cla@nvidia.com](mailto:torch-dgx-cla@nvidia.com), as applicable.
- [ ] New or existing tests cover these changes, or I explained why tests are not applicable.
- [ ] The documentation is up to date with these changes.
