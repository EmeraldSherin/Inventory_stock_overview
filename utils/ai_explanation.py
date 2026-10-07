"""Reusable AI Insight box for the dashboard pages.

Each page passes a small dictionary of already-calculated summary values.

If a Gemini API key is configured (Streamlit secrets or environment
variable GEMINI_API_KEY), Google's Gemini model writes a short explanation
from that dictionary only.

Otherwise - or if the call fails, times out, or returns text containing
numbers that are not in the supplied context - a deterministic rule-based
explanation built from the same values is shown instead.
"""

import json
import logging
import math
import os
import re
import time

from google import genai
import streamlit as st


DEFAULT_MODEL = "gemini-2.5-flash"
FAILURE_COOLDOWN_SECONDS = 60

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Gemini system prompt
# -------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an inventory analytics explanation assistant. "
    "Explain the dashboard results in simple business language. "

    "Use ONLY the metrics and facts provided in the input context. "
    "Do not invent values or facts. "

    "Never invent numbers, products, forecasts, probabilities, savings, "
    "thresholds, or business facts. "

    "Use the supplied numbers accurately. "
    "Do not round, convert, estimate, or derive new numbers. "

    "Do not claim that a calculation is AI or machine learning unless "
    "the provided context explicitly identifies it as such. "

    "Identify the 1-3 most important insights instead of repeating "
    "every metric. "

    "Keep the response concise at approximately 2-4 complete sentences. "
    "Always return a complete response and never stop mid-sentence. "

    "Do not use markdown tables. "

    "Do not provide generic advice unrelated to the supplied data. "

    "If a value is not supplied, do not mention it."
)


# -------------------------------------------------------------------
# Page-specific focus
# -------------------------------------------------------------------

PAGE_FOCUS = {

    "executive_dashboard": (
        "What is the most important thing happening in the current "
        "dashboard? Cover the overall business situation including "
        "sales, profit, and inventory position."
    ),

    "demand_forecast": (
        "What is happening with demand? Talk only about demand, "
        "the baseline forecast, demand trend, and promotions. "
        "Do not mention inventory, replenishment, or forecast error metrics."
    ),

    "inventory_risk": (
        "What is wrong with the current inventory? "
        "Talk only about current inventory risk. "
        "Below Reorder Point is not a stockout. "
        "If critical or stockout counts are zero, say none are observed."
    ),

    "replenishment": (
        "What should be paid attention to or replenished? "
        "Talk only about the replenishment results supplied. "
        "Do not add recommendations that are not in the data."
    ),
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _clean(value):
    """Convert numpy/pandas values to plain Python."""

    if isinstance(value, dict):
        out = {k: _clean(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v is not None}

    if isinstance(value, (list, tuple)):
        return [
            v for v in (_clean(v) for v in value)
            if v is not None
        ]

    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            value = str(value)

    if isinstance(value, float):
        return (
            None
            if (math.isnan(value) or math.isinf(value))
            else round(value, 2)
        )

    if isinstance(value, (int, str, bool)) or value is None:
        return value

    return str(value)


# Correct number pattern
_NUM = re.compile(r"\d[\d,]*\.?\d*")


def _numbers(text):
    """Extract numeric values from text."""

    out = []

    for m in _NUM.findall(text):
        m = m.rstrip(".,").replace(",", "")

        try:
            out.append(float(m))
        except ValueError:
            pass

    return out


def _numbers_supported(text, context_json):
    """Return True if every meaningful number is present in the context."""

    allowed = _numbers(context_json)

    for n in _numbers(text):

        # Small sentence numbers such as 1, 2, 3 are allowed.
        if n <= 3:
            continue

        if not any(
            abs(n - a) <= max(0.051, 0.002 * abs(a))
            for a in allowed
        ):
            return False

    return True


def _get_secret(name):
    """Read a value from Streamlit secrets or environment variables."""

    try:
        value = st.secrets.get(name)

        if value:
            return str(value)

    except Exception:
        pass

    return os.environ.get(name) or None


def _get_api_key():
    return _get_secret("GEMINI_API_KEY")


def _get_model():
    return _get_secret("GEMINI_MODEL") or DEFAULT_MODEL


# -------------------------------------------------------------------
# LLM explanation
# -------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False, max_entries=200)
def _llm_explanation(page: str, context_json: str, model: str) -> str:
    """Generate a short Gemini explanation from dashboard data only."""

    api_key = _get_api_key()

    if not api_key:
        raise RuntimeError("No Gemini API key")

    client = genai.Client(api_key=api_key)

    prompt = (
        f"Question to answer:\n"
        f"{PAGE_FOCUS[page]}\n\n"

        f"Dashboard data (JSON):\n"
        f"{context_json}\n\n"

        "Use only the supplied dashboard data. "
        "Do not invent values or facts. "

        "Return a complete 2-4 sentence explanation. "
        "Do not stop mid-sentence. "

        "Use only numbers that appear in the supplied dashboard data. "
        "Do not create additional numbers."
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.2,
            "max_output_tokens": 1500,
        },
    )

    text = (response.text or "").strip()

    if not text:
        raise ValueError("Gemini returned an empty response")

    if len(text) > 1200:
        raise ValueError("Gemini response is too long")

    if not _numbers_supported(text, context_json):
        raise ValueError(
            "Gemini response contained numbers not present in the context"
        )

    return text


# -------------------------------------------------------------------
# Rule-based fallback
# -------------------------------------------------------------------

def _fmt(x, d=0):
    return f"{x:,.{d}f}"


def _fallback_executive(c):

    parts = []

    if all(
        k in c
        for k in (
            "total_revenue",
            "gross_profit",
            "gross_margin_pct",
            "units_sold",
        )
    ):
        parts.append(
            f"The selected period generated revenue of "
            f"{_fmt(c['total_revenue'])} and gross profit of "
            f"{_fmt(c['gross_profit'])} "
            f"(gross margin {_fmt(c['gross_margin_pct'], 1)}%) "
            f"on {_fmt(c['units_sold'])} units sold."
        )

    if "top_region" in c and "top_region_revenue_share_pct" in c:
        parts.append(
            f"{c['top_region']} contributes the largest share of revenue "
            f"({_fmt(c['top_region_revenue_share_pct'], 1)}%)."
        )

    if "below_reorder_count" in c and "inventory_items" in c:

        if c["below_reorder_count"] == 0:

            s = (
                f"In the current inventory snapshot, none of the "
                f"{c['inventory_items']} SKU-Warehouse combinations "
                f"are below their reorder point"
            )

        else:

            s = (
                f"In the current inventory snapshot, "
                f"{c['below_reorder_count']} of "
                f"{c['inventory_items']} SKU-Warehouse combinations "
                f"({_fmt(c.get('below_reorder_rate_pct', 0), 1)}%) "
                f"are below their reorder point"
            )

            if "highest_risk_warehouse" in c:
                s += (
                    f", with {c['highest_risk_warehouse']} showing "
                    f"the highest share "
                    f"({_fmt(c['highest_risk_warehouse_rate_pct'], 1)}%)"
                )

        parts.append(s + ".")

    return " ".join(parts)


def _fallback_demand(c):

    parts = []

    if "average_demand" in c and "peak_demand" in c:

        s = (
            f"Average demand is "
            f"{_fmt(c['average_demand'], 2)} units per "
            f"SKU-Warehouse per day, with a peak of "
            f"{_fmt(c['peak_demand'])}"
        )

        if "demand_variability_cv" in c:
            s += (
                f" and a variability (CV) of "
                f"{_fmt(c['demand_variability_cv'], 2)}"
            )

        parts.append(s + ".")

    if "trend_pct" in c:

        direction = (
            "higher"
            if c["trend_pct"] > 0
            else "lower"
            if c["trend_pct"] < 0
            else "unchanged"
        )

        if direction == "unchanged":

            parts.append(
                "Demand over the last 30 days is unchanged "
                "versus the prior 30 days."
            )

        else:

            parts.append(
                f"Demand over the last 30 days is "
                f"{_fmt(abs(c['trend_pct']), 1)}% "
                f"{direction} than the prior 30 days."
            )

    if (
        "avg_demand_promotion" in c
        and "avg_demand_no_promotion" in c
    ):

        rel = (
            "higher"
            if c["avg_demand_promotion"]
            > c["avg_demand_no_promotion"]
            else "lower"
        )

        parts.append(
            f"Promotion days average "
            f"{_fmt(c['avg_demand_promotion'], 2)} units versus "
            f"{_fmt(c['avg_demand_no_promotion'], 2)} without "
            f"promotion ({rel} on promotion)."
        )

    if "average_forecast_demand" in c and "average_demand" in c:

        parts.append(
            f"The baseline forecast averages "
            f"{_fmt(c['average_forecast_demand'], 2)} units per day "
            f"against actual demand of "
            f"{_fmt(c['average_demand'], 2)}."
        )

    return " ".join(parts)


def _fallback_inventory(c):

    parts = []

    if "below_reorder_count" in c and "inventory_items" in c:

        if c["below_reorder_count"] == 0:

            parts.append(
                f"None of the {c['inventory_items']} "
                f"SKU-Warehouse combinations are below their "
                f"reorder point in the current snapshot."
            )

        else:

            parts.append(
                f"{c['below_reorder_count']} of "
                f"{c['inventory_items']} SKU-Warehouse combinations "
                f"({_fmt(c.get('below_reorder_pct', 0), 1)}%) "
                f"are below their reorder point."
            )

    if "critical_count" in c:

        if (
            c["critical_count"] == 0
            and c.get("stockout_count", 0) == 0
        ):

            parts.append(
                "No critical inventory or stockouts are observed "
                "in the current snapshot."
            )

        else:

            parts.append(
                f"{c['critical_count']} items are critical and "
                f"{c.get('stockout_count', 0)} are stocked out."
            )

    if c.get("most_at_risk"):

        m = c["most_at_risk"][0]

        if m["distance_to_reorder_point"] < 0:

            parts.append(
                f"The item furthest below its reorder point is "
                f"{m['item']} "
                f"(inventory {_fmt(m['inventory_level'])} "
                f"vs reorder point {_fmt(m['reorder_point'])})."
            )

    if (
        "highest_risk_warehouse" in c
        and c.get("below_reorder_count", 0) > 0
    ):

        parts.append(
            f"{c['highest_risk_warehouse']} has the most items "
            f"below reorder point "
            f"({c['highest_risk_warehouse_count']})."
        )

    return " ".join(parts)


def _fallback_replenishment(c):

    parts = []

    if "items_to_replenish" in c and "inventory_items" in c:

        if c["items_to_replenish"] == 0:

            parts.append(
                "No replenishment is required for the current "
                "filtered inventory snapshot."
            )

        else:

            parts.append(
                f"{c['items_to_replenish']} of "
                f"{c['inventory_items']} SKU-Warehouse combinations "
                f"need replenishment, totalling "
                f"{_fmt(c.get('recommended_units', 0))} units "
                f"at an estimated cost of "
                f"{_fmt(c.get('replenishment_cost', 0))}."
            )

    if c.get("high_priority_count", 0) > 0:

        parts.append(
            f"{c['high_priority_count']} of them are high priority, "
            f"where inventory is critical or below expected "
            f"lead-time demand."
        )

    if c.get("top_items"):

        t = c["top_items"][0]

        parts.append(
            f"The largest recommended quantity is "
            f"{_fmt(t['recommended_quantity'])} units for "
            f"{t['item']} ({t['priority']} priority)."
        )

    if c.get("monitor_count", 0) > 0:

        parts.append(
            f"A further {c['monitor_count']} items are on Monitor "
            f"but need no order yet."
        )

    return " ".join(parts)


_FALLBACKS = {
    "executive_dashboard": _fallback_executive,
    "demand_forecast": _fallback_demand,
    "inventory_risk": _fallback_inventory,
    "replenishment": _fallback_replenishment,
}


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def generate_ai_explanation(page: str, context: dict):
    """Return (text, source) where source is 'llm' or 'rule-based'."""

    ctx = _clean(context)

    fallback = (
        _FALLBACKS[page](ctx)
        or "No summary is available for the current selection."
    )

    api_key = _get_api_key()

    if not api_key:
        return fallback, "rule-based"

    # After a failure, skip the API briefly instead of retrying
    # on every Streamlit rerun.
    if (
        time.time()
        - st.session_state.get("_ai_last_failure", 0)
        < FAILURE_COOLDOWN_SECONDS
    ):
        return fallback, "rule-based"

    context_json = json.dumps(
        ctx,
        sort_keys=True,
        default=str,
    )

    try:

        return (
            _llm_explanation(
                page,
                context_json,
                _get_model(),
            ),
            "llm",
        )

    except Exception as exc:

        st.session_state["_ai_last_failure"] = time.time()

        logger.warning(
            "Gemini explanation failed (%s); using rule-based fallback.",
            type(exc).__name__,
        )

        return fallback, "rule-based"


def render_ai_insight(page: str, context: dict):
    """Render the small insight box. Never raises."""

    try:

        text, source = generate_ai_explanation(
            page,
            context,
        )

    except Exception:

        return

    with st.container(border=True):

        if source == "llm":

            st.markdown("**🤖 AI Insight**")
            st.write(text)
            st.caption(
                "Generated by an LLM from the figures shown on this page."
            )

        else:

            st.markdown("**📊 Dashboard Insight**")
            st.write(text)
            st.caption(
                "Rule-based summary of the figures shown on this page "
                "(LLM not used)."
            )