#!/usr/bin/env python
"""Back-compat shim. The runner moved into ``emboviz._internal.runner``
when the consolidated ``emboviz`` console command was introduced.

Prefer ``emboviz analyze`` for new work. This script is kept so that
``scripts/final_integration_test.sh`` and other pre-existing setup
scripts keep working without modification.
"""
from emboviz._internal.runner import main

if __name__ == "__main__":
    main()
