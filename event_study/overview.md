# Event Study Overview

I am doing an Advanced NLP project where I am trying to do a multimodal analysis of corporate earnings conference calls (audio and transcript) to obtain a trading signal for the companies shares. The idea is to do sentiment analysis of the text transcript as well as the conference call audio data and see if it correlates with the stock price.

I have downloaded the dataset of 571 earnings calls that is available at `https://github.com/GeminiLn/EarningsCall_Dataset`. The audio data has been placed at `~/var/data/earnings_calls/ACL19_Release`.

I have also used Yahoo Finance and downloaded the daily Stock Price history of the companies from a little bit before the time of the call to 6 months after it.

The code for this download is in the directory `../acl19_proc` and the price data is in the subdirectory `../acl19_proc/data`.

I think that the next step is to try and determine if there has been any change in the stock price from before the call to after it. 

I think that this is called an **Event Study** — it's the canonical finance methodology for exactly this (Fama, Fisher, Jensen & Roll 1969; MacKinlay's 1997 survey is the standard reference). The core idea: measure **Abnormal Returns (AR)** — what the stock did versus what it *should* have done given the market — rather than raw price changes.

## The Methodology

**1. Define the event date carefully.** Most earnings calls happen after market close. If a call is at 5pm on Tuesday, the price reaction lands on Wednesday's open. If your dataset has call timestamps, use them; otherwise, assume next-trading-day as day 0 and sanity-check robustness with both conventions. Getting this wrong by one day will smear your entire effect.

**2. Estimate "normal" returns.** For each event, fit a model over a clean estimation window, e.g. trading days −150 to −30 before the call:

- **Market model** (standard): regress stock returns on market returns (use SPY or ^GSPC): `R_i = α + β·R_m + ε`
- **Market-adjusted** (simpler, less powerful): expected return = market return
- Fama-French 3/5-factor if you want to be fancy, but market model is fine at this scale

**3. Compute abnormal returns over the event window:**

`AR_t = R_t(actual) − E[R_t](predicted)`

`CAR(τ₁, τ₂) = Σ AR_t` over the window

Run multiple windows:
- **[0, +1]** — immediate reaction (the "surprise")
- **[0, +20]** / **[0, +60]** — captures PEAD (Post-Earnings Announcement Drift), the well-documented tendency of stocks to keep drifting in the direction of the surprise for weeks. This is where a tradeable signal actually lives — the overnight jump is already priced by the time you could trade it.

**4. Test significance.** Average the CARs across your 571 events, t-test against zero. Then the real question: cross-sectional regression of CAR on your sentiment scores.

## Gotchas

- **Use Adj Close** from Yahoo — otherwise splits/dividends will look like crashes.
- **Earnings release ≠ call.** The press release usually drops hours before the call, so the day-0 reaction bundles both the numbers and the tone. You can't cleanly separate them without release timestamps — but for your purposes that's fine: your signal is "call + numbers bundle."
- **Event clustering.** Earnings season means many of the 571 events land in the same weeks, so returns are cross-sectionally correlated → inflated t-stats. Fine for a course project, worth a one-line caveat in the writeup.
- 571 calls across presumably ~100+ companies means some firms appear multiple times — not independent observations, same caveat bucket.

## Framing for the project

The event study isn't really the research question — it's my **evaluation harness**. The actual question is: *does multimodal sentiment predict CAR?* Concretely: bucket the 571 events by sentiment score, plot mean CAR curves per bucket, and check whether high-sentiment calls show positive drift and low-sentiment negative. A monotonic separation between buckets is the money shot. (Bonus: does *audio* sentiment add predictive power over transcript sentiment alone? That's your multimodal claim.)

