"""Soccer Vision: evidence-linked soccer video analytics.

Correctness invariants (enforced throughout the package):
  1. Observations are not verified events.
  2. Tracking IDs are not roster identities.
  3. Missing observations are not zeros.
  4. No unreviewed prediction appears as a confirmed player statistic.
"""

__version__ = "0.1.0"
