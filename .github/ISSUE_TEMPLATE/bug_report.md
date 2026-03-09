---
name: Bug Report
about: Something isn't working as expected
title: "[Bug] "
labels: bug
assignees: ''
---

## Describe the Bug
A clear description of what the bug is.

## To Reproduce

```python
# Minimal reproduction case
from rag_doctor import Doctor

doctor = Doctor.default()
report = doctor.diagnose(
    query="...",
    answer="...",
)
```

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened. Include the full error traceback if applicable.

## Environment

- Python version: [e.g. 3.11]
- rag-doctor version: [e.g. 1.0.0]
- OS: [e.g. Ubuntu 22.04]

## Additional Context
Any other context about the problem.
