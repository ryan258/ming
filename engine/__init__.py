"""Core runtime for the Ming agent harness.

Import concrete classes from their modules. Keeping package initialization inert prevents the
capability index and orchestrator from forming an import-order-dependent cycle.
"""
