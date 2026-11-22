"""DEPRECATED: Standalone Streamlit UI retired in favour of the Unified Portal.

Please launch the unified Next.js Executive Portal in:
  `brand-analytics-agent/frontend-portal/` (Velocity & AI Ops page `/velocity`)
backed by `GET /api/v1/portal/velocity`, `data/gold_velocity/velocity_scores.parquet`,
and `data/gold_velocity/drift_report.json`.
"""

from __future__ import annotations

import warnings

DEPRECATION_MESSAGE = (
    "competitive-velocity-ml-engine/app/streamlit_app.py is deprecated and retired. "
    "Use brand-analytics-agent/frontend-portal (/velocity) instead."
)


def main() -> None:
    warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
    raise RuntimeError(DEPRECATION_MESSAGE)


if __name__ == "__main__":
    main()
