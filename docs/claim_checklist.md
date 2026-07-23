# Claim Strength Checklist

Before writing any sentence containing a factual claim, classify it:

## Class A: Empirically Verified (in this paper)
- GSM8K trajectory and Δ_min (§3) — all specific numbers are Class A
- Trap documentation (§4) — the events happened as described
- These can use direct, unqualified language. Examples: "ρ_trend = -0.091",
  "Δ_min = 9.8% at N=200", "the proxy signal was later found to be synthetic"

## Class B: Logical Inference (not directly verified here)
- Section 5.2 point 2 (noise-robust aggregation preferred) — logically follows
  from Class A but not empirically tested
- Section 5.1 last sentence (PAAS may be relevant in other settings)
- Must use qualifying language: "it is a logical consequence", "we did not test
  this", "suggested by", "may be"

## Class C: Acknowledged Limitations
- Section 6 entries
- Statements about what the paper does NOT claim
- Can use direct language but should explicitly label as limitations

## Rules for Writing
1. Every Class A claim must be traceable to a specific table/figure/section
   in this document
2. Every Class B claim must be adjacent to a qualification like
   "we did not empirically verify this"
3. Class B claims must never be mixed with Class A claims in the same
   sentence — it creates ambiguity about which part is verified
4. Introduction/Abstract may only use Class A claims + one sentence about
   the paper's nature (diagnostic, not algorithmic)
