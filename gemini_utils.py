"""
Shared Gemini helpers: transient-error retries and cross-model fallback.

Used by main.py, editor.py, buffer_scheduler.py, and saasshorts.py so they all
share one model preference list and one resilience policy.
"""

import time


# Gemini models in preference order. If the primary model is unavailable (not
# yet released, no access, or 404'd) — or exhausts its transient retries — callers
# transparently fall back to the next one so the pipeline keeps working.
#
# Tiers are matched to the job:
#   - GEMINI_PRO_MODELS  — hard judgment calls made ONCE per episode (viral clip
#     selection). Pro's reasoning + instruction-following is worth the cost when
#     the call is rare and high-leverage.
#   - GEMINI_MODELS      — the shared default (editor effects, scheduling, etc.):
#     a capable, stable Flash tier.
#   - GEMINI_LITE_MODELS — trivial, high-frequency calls (b-roll cue planning).
#     Lite is plenty for "name a visualizable noun" and ~6x cheaper than Flash.
#
# Every tier falls back to gemini-3.5-flash as a stable safety net.
GEMINI_PRO_MODELS = ("gemini-3.1-pro-preview", "gemini-3.5-flash")
GEMINI_MODELS = ("gemini-3.5-flash", "gemini-3.1-flash-lite")
GEMINI_LITE_MODELS = ("gemini-3.1-flash-lite", "gemini-3.5-flash")


def _is_transient_gemini_error(exc):
    """True for errors worth retrying (overload/rate-limit/timeout), not permanent ones."""
    msg = str(exc).upper()
    transient_markers = (
        "503", "UNAVAILABLE",        # model overloaded / high demand
        "429", "RESOURCE_EXHAUSTED", # rate limited
        "500", "INTERNAL",           # transient server error
        "504", "DEADLINE", "TIMEOUT", "TIMED OUT",
        "CONNECTION", "TEMPORARILY",
    )
    return any(marker in msg for marker in transient_markers)


def _is_model_unavailable_error(exc):
    """True when a specific model can't be used at all (vs. a transient blip).

    Retrying the same model won't help these, so the caller should fall back to
    the next model in the list instead.
    """
    msg = str(exc).upper()
    markers = (
        "404", "NOT_FOUND", "NOT FOUND",
        "DOES NOT EXIST", "IS NOT SUPPORTED", "NOT SUPPORTED",
        "403", "PERMISSION_DENIED",  # no access to (preview) model
    )
    return any(marker in msg for marker in markers)


def gemini_generate_with_fallback(client, contents, models=GEMINI_MODELS,
                                  max_attempts=5, **kwargs):
    """Call Gemini, retrying transient errors and falling back across models.

    Transient errors (overload/rate-limit/timeout) retry the same model with
    exponential backoff; if they persist through every attempt, we fall back to
    the next model (its per-model quota is separate, so the fallback can succeed
    where the primary is exhausted). Model-unavailable errors skip straight to
    the next model. Any extra kwargs (e.g. ``config``) are forwarded to
    ``generate_content``.

    Returns (response, model_name) on success, or (None, None) on hard failure.
    """
    last_exc = None
    for model_name in models:
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    **kwargs
                )
                return response, model_name
            except Exception as e:
                last_exc = e
                if _is_transient_gemini_error(e):
                    if attempt < max_attempts:
                        delay = min(2 ** attempt, 30)  # 2,4,8,16s ... capped at 30
                        print(f"⚠️  Gemini transient error on {model_name} "
                              f"(attempt {attempt}/{max_attempts}): {e}")
                        print(f"   Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    # Retries exhausted on this model — fall back to the next one
                    # rather than giving up (a separate model quota may be free).
                    print(f"⚠️  Gemini transient error on {model_name} persisted "
                          f"through {max_attempts} attempts: {e}")
                    print(f"   Falling back to next model...")
                    break
                if _is_model_unavailable_error(e):
                    print(f"⚠️  Model '{model_name}' unavailable: {e}")
                    print(f"   Falling back to next model...")
                    break  # stop retrying this model, try the next one
                print(f"❌ Gemini Error on {model_name}: {e}")
                return None, None
    print(f"❌ All Gemini models exhausted. Last error: {last_exc}")
    return None, None
