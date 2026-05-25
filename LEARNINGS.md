# LEARNINGS.md — v1 to v2 Growth Journey

This file documents the specific things that went wrong, the things that
surprised me, and the decisions I changed between the first version of this
project and the second.

---

## Why There's a v2 at All

Prof. Johan Loeckx reviewed the first notebook and told me it looked AI-generated.
He was right to say it. The cells ran perfectly from top to bottom with no
hesitation, no wrong turns, no moments of "I wasn't sure about this." That's
not how analysis actually goes. v2 is the honest version.

---

## Mistake 1: Data Leakage in the First Pipeline

The first version fit StandardScaler on the full dataset before the train/test
split. Everything looked fine. Cross-validation accuracy was around 83%.

Then I read about leakage properly. The scaler had computed mean and standard
deviation from all 891 rows — including the 179 I was using as a test set.
The test set was technically "known" to the preprocessing step. The scores were
slightly contaminated.

Rebuilding with Pipeline — where the scaler fits only on the training fold
during each CV iteration — dropped me to about 81% hold-out accuracy. That 2%
was borrowed from the future.

The fix is mechanical once you understand it. The learning is understanding why
it matters in practice, not just in principle.

---

## Mistake 2: GridSearchCV Scoring on Accuracy

First version used `scoring='accuracy'` in GridSearchCV. The DummyClassifier
that always predicts "died" was hitting 62% — barely 15 points below my tuned
model. That gap looked big until I thought about it: a 62% baseline for free
means I only have 38 points of room to improve, and I was using 15 of them.

Switching to `scoring='f1'` changed the optimal hyperparameters (C dropped
from 1.0 to 0.5) and made the comparison against the dummy genuinely meaningful.
F1 penalises the kind of lazy prediction that accuracy rewarded.

---

## Mistake 3: Treating the Threshold as Fixed

I had never questioned the 0.50 decision threshold. It felt like a law.

The threshold is a policy choice. In an evacuation problem, failing to identify
a survivor who needed help (false negative) is worse than flagging someone who
would have been fine anyway (false positive). F2 score weights recall more heavily
than precision. The F2-optimal threshold turned out to be 0.31, not 0.50.

What this means practically: at 0.31, the model catches more survivors at the
cost of more false alarms. Whether that's the right tradeoff depends on what
you're building. The point is that 0.50 is never the automatic answer.

---

## Surprise: The Size of the Sex=female Coefficient

I expected Sex to matter. I did not expect the magnitude.

Sex=female came out at +2.61 in standardised log-odds. exp(2.61) ≈ 13.5.
A female passenger had about 13.5 times higher odds of surviving than a
comparable male passenger. That is not a subtle effect. It is the dominant
signal in the dataset by a large margin.

The "women and children first" protocol wasn't just a cultural norm — it was
enforced so consistently that it produced a 13.5 odds ratio in the data.
Thirteen-point-five. That stopped me.

---

## Decision I Would Revisit: Age Imputation

I used group-aware median imputation (filling missing Age with the median for
each Pclass × Sex group). This is better than a single overall median, and the
group differences were large enough to justify it.

But it's still a simplification. A proper approach would use a regression model
to predict Age from all other available features. I didn't do that because it
would have added significant complexity to the pipeline for a marginal improvement
in a variable that isn't the most important predictor anyway.

If I were putting this into production for real decisions, I would revisit this.

---

## Decision I Would Revisit: Correlated Features

FamilySize, SibSp, and Parch are highly correlated (FamilySize is literally
defined as SibSp + Parch + 1). I kept all three, reasoning that L2 regularisation
handles multicollinearity by shrinking correlated coefficients.

I did not verify this by running the model both with and without the redundant
features and comparing AUC. I argued from theory but didn't test the claim.
That's a gap. The experiment would take ten minutes.

---

## On Building This from Android

Termux on Android: numpy refused to compile from source on ARM64.
The error message wasn't in any Stack Overflow answer I could find.
The fix: `pkg install python-numpy` before anything else, then
`pip install --break-system-packages` for the remaining packages.
Two hours of reading to produce one working command.

Colab via mobile browser: session timeouts are real and they cost you.
I learned to write defensive code — save model outputs to disk at every
checkpoint. If the session dies, you don't restart from the beginning.

The constraint changed how I think about environment setup. A pipeline that
works on a $150 Android phone with bad internet is more production-ready than
one that requires a specific conda environment and a stable connection.

---

## What v2 Is and Isn't

v2 is not a better model than v1 in terms of numbers. The metrics are similar.

v2 is a more honest account of how the model was built — what I got wrong,
what I didn't expect, what I would do differently. If someone reads this project
and asks me to explain any decision, I can explain it, including the decisions
I'm uncertain about.

That's what the professor was asking for.

---
*James Koero — Kisumu, Kenya — May 2026*
