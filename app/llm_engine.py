"""
LLM integration — generates business-friendly summaries of causal analysis results.
Supports Anthropic, OpenAI, and Jan AI (local). Auto-detects provider from .env.
"""
import os
import json
import urllib.request


def _build_prompt(results):
    ov = results["overview"]
    psm = results["psm"]
    upl = results["uplift"]

    segments = "\n".join(
        f"  {s['quartile']} (n={s['n']:,}): predicted uplift {s['avg_uplift_pct']:+.1f}pp | "
        f"treated {s['treatment_conversion']:.1%} vs control {s['control_conversion']:.1%} | "
        f"{s['recommendation']}"
        for s in upl["segment_stats"]
    )

    return f"""You are a senior data scientist presenting causal inference results to a business leadership team.

EXPERIMENT: An online advertising incrementality trial — the platform randomly decides whether to bid on an ad impression for each user (treatment) or withhold the bid (control).
QUESTION: Does showing the ad cause more site visits?

DATASET: Criteo Uplift Modeling Dataset (Diemert et al., 2018)
- Sample size: {ov['n_total']:,} users (full Criteo dataset)
- Ad shown (treated): {ov['n_treated']:,}
- No ad (control): {ov['n_control']:,}
- Overall visit rate: {ov['overall_conversion']:.2%}
- Naive ATE (biased): {ov['naive_ate_pct']:+.2f}pp

PROPENSITY SCORE MATCHING:
- Matched pairs: {psm['n_matched']:,}
- Control visit rate: {psm['control_conversion_rate']:.2%}
- Treated visit rate: {psm['treated_conversion_rate']:.2%}
- ATE (adjusted): {psm['ate_pct']:+.2f}pp
- 95% CI: [{psm['ci_low']*100:+.2f}pp, {psm['ci_high']*100:+.2f}pp]
- P-value: {psm['p_value']:.6f} ({'SIGNIFICANT' if psm['significant'] else 'NOT significant'})

UPLIFT MODEL — CATE BY SEGMENT:
{segments}

Write a business summary in 4 short paragraphs:
1. KEY FINDING: headline result, compare naive vs adjusted ATE, explain why they differ (or confirm they align if the trial was well-randomized).
2. WHY CAUSAL INFERENCE: why simple comparison can be misleading even in RCTs with imperfect compliance, what PSM validates.
3. WHO RESPONDS BEST: interpret quartiles, which user segments to target and which to skip, with numbers.
4. RECOMMENDATIONS: 3 bullet points, concrete and actionable for ad spend optimization.

Rules: plain English, include numbers, be direct, ~280 words total."""


def _post_openai_compatible(url, model, prompt, api_key=""):
    """Generic OpenAI-compatible chat completion request."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700,
        "temperature": 0.3,
        "stream": False,
    }).encode()

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode())
    return body["choices"][0]["message"]["content"].strip()


def _call_anthropic(prompt):
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5").strip()
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model,
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _detect_provider():
    # explicit override
    explicit = os.getenv("LLM_PROVIDER", "").lower().strip()
    if explicit:
        return explicit
    # auto-detect from available keys
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    return "jan"


def generate_causal_summary(results):
    provider = _detect_provider()
    prompt = _build_prompt(results)

    if provider == "jan":
        url = os.getenv("JAN_URL", "http://127.0.0.1:1337/v1/chat/completions")
        model = os.getenv("JAN_MODEL", "llama3.2-3b-instruct")
        print(f"[llm] using Jan AI local, model={model}")
        return _post_openai_compatible(url, model, prompt)

    elif provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        print(f"[llm] using Anthropic, model={model}")
        return _call_anthropic(prompt)

    elif provider == "openai":
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        url = base + "/chat/completions"
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        key = os.getenv("OPENAI_API_KEY", "").strip()
        print(f"[llm] using OpenAI-compatible, model={model}")
        return _post_openai_compatible(url, model, prompt, api_key=key)

    else:
        raise RuntimeError(
            f"Unknown LLM provider '{provider}'. Set LLM_PROVIDER to: jan | anthropic | openai"
        )
