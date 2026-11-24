# Strategy explanation — Algo Arena

## One-paragraph version

The market is simulated, so instead of fitting a black box we worked out how the simulator makes prices and predicted from that. Every headline pushes the price at a steady rate for the next 10 bars and then stops. At the close of bar 49, headlines from bars 40–49 have not finished pushing yet, so the return over bars 50–59 is partly already decided. We add up that unfinished push, count only the headlines that actually matter for this stock, and take bigger positions where the forecast is strong relative to the session's noise.

## What we found in the data

1. **Headlines are templates.** 80 companies and 50 kinds of event once numbers, regions and segments are stripped out. The wording carries no information beyond the kind of event — sentiment-style text models do badly here.
2. **Each kind of event has a fixed effect with a fixed shape.** An event study on the 10,000 labelled sessions shows the price drifting at a constant rate over the 10 bars after a headline (half as much in the bar it is published and in the 10th bar after), and nothing before or after. Some kinds are strongly positive, some strongly negative, about 20 do nothing.
3. **Not every headline in a session moves that session's stock.** Companies fall into 5 sectors of 16. Every session contains 3 companies from one sector plus 1 from anywhere. Headlines about the odd-sector company have no measurable effect.
4. **One of the three same-sector companies is the stock itself.** Its own headlines move the price at full strength; the other two move it at about a quarter of that. Which company it is isn't given — it has to be inferred from how the price reacted to earlier headlines.

## The forecast

- **Unfinished push.** For each headline in bars 40–49, effect of its event kind × the share of its 10-bar push that falls in bars 50–59.
- **Who is the stock?** For each candidate company, compare the first 50 bars of returns with what its headlines should have done. That gives a probability for each candidate (including "the stock hasn't appeared in a headline yet"). The forecast weights each headline by the chance it is about the stock itself (full weight) or a same-sector peer (quarter weight). With only 50 noisy bars this probability is uncertain, so it is deliberately softened rather than trusted outright.
- **Small corrections** in the second model: sessions with a lot of early news tend to have less later, repeated news about one company, conflicting news, and how much of the first-half move the headlines fail to explain.

## Position sizing

The score is the Sharpe ratio of `position × return`, so only relative size across sessions matters. For a forecast μ and return variance σ², the Sharpe-maximising position is proportional to **μ / (σ² + μ²)**: bet more when the forecast is large relative to the session's noise, and stop scaling up once the forecast is large in absolute terms. Each session's noise is estimated from its own first 50 bars using the high–low range of every bar (Garman–Klass), which is far less noisy than close-to-close moves and was worth +0.05 Sharpe on its own.

## Final submission: a blend of two models

Both use the structure above but estimate it differently, so their errors differ.

| Model | How it works | Cross-validated Sharpe |
|---|---|---|
| A. Regression model | Unfinished push from main-sector headlines plus the small corrections, linear fit, positions divided by predicted risk. Built on a publicly available solution to this challenge. | 3.47 |
| B. Hidden-company model | Probability model over which company is the stock, fitted by expectation–maximisation; forecast is the probability-weighted unfinished push; Sharpe-optimal sizing with a range-based volatility estimate. Our own. | 3.49 |
| **50/50 blend** (each scaled to unit size first) | | **3.54** |

The two are 93% correlated. On identical folds the blend beats model A alone by +0.06 to +0.07 Sharpe, positive in over 99% of bootstrap resamples.

## How we checked it

- 5-fold cross-validation on the labelled sessions, repeated with two different fold assignments. Parameters are fitted on the training folds only; the prediction step is physically given only bars 0–49 and their headlines.
- Models compared on the same sessions, not on two separate headline numbers — one 10,000-session Sharpe has a standard error of about 0.15.
- Independent review of the code for look-ahead before any upload.

## What did not help

Sentiment and word-level text features, the numbers inside headlines, gradient boosting on the same inputs, past price momentum, and trusting the "which company" probability at face value (it is over-confident on 50 bars).
