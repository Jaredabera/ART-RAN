# Contributing to ART-RAN

Thank you for your interest in ART-RAN.

For reproducibility or bug fixes, please open an issue describing the affected script, configuration, dataset split, seed, and expected/observed behavior. Pull requests should be focused and should not commit raw COMMAG data, virtual environments, generated model binaries, or large training logs.

Before submitting a pull request:

```bash
python -m compileall art_ran_commag scripts
```

If tests are available in the tagged release, run them as well.

Research extensions should clearly distinguish changes to the threat model, rApp reasoner, Near-RT controller, dataset preprocessing, and evaluation protocol.
