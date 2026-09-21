"""Member 2 — classifier, SHAP explainability, evidence weighting, fusion.

This package is deliberately NOT named `app`: it runs alongside Member 1's
`app` package (this same project's app/), importing M1's DB models, config and
Celery app rather than duplicating them. Two packages both named `app`
would shadow each other on sys.path.
"""
