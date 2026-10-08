# IV) Results – Wine 90+ Classification

Full Wine Data.xlsx: 80/20 stratified split (seed 42); 5-fold cross-validation on all 1010 wines.

## A) Naive Bayes

- Test accuracy **87.13%**, precision 93.18%, recall 80.39%, specificity 94.00%, F1 86.32%
- 5-fold CV accuracy 87.82% ± 1.35%
- Figures: A1–A4

## B) Decision Tree (depth 5–9)

| Depth | Train acc | Test acc | Precision | Recall | F1 | CV mean | Leaves |
|---|---|---|---|---|---|---|---|
| 5 | 77.10% | 70.30% | 86.21% | 49.02% | 62.50% | 74.55% | 16 |
| 6 | 80.07% | 71.78% | 84.62% | 53.92% | 65.87% | 76.14% | 22 |
| 7 | 82.55% | 74.26% | 85.71% | 58.82% | 69.77% | 75.94% | 30 |
| 8 | 84.41% | 75.25% | 86.11% | 60.78% | 71.26% | 78.02% | 35 |
| 9 | 86.01% | 72.77% | 82.19% | 58.82% | 68.57% | 78.32% | 41 |

Best depth by cross-validation: **9**. Figures: B1–B5

## C) SVM (linear kernel)

- Test accuracy **86.63%**, precision 89.47%, recall 83.33%, specificity 90.00%, F1 86.29%
- 5-fold CV accuracy 86.53% ± 3.44%
- Support vectors: 282
- Figures: C1–C4

## D) Comparison

| Model | Accuracy | Precision | Recall | Specificity | F1 | AUC | 5-fold CV |
|---|---|---|---|---|---|---|---|
| Naive Bayes | 87.13% | 93.18% | 80.39% | 94.00% | 86.32% | 0.944 | 87.82% ± 1.35% |
| Decision Tree (depth 9) | 72.77% | 82.19% | 58.82% | 87.00% | 68.57% | 0.740 | 78.32% ± 3.09% |
| SVM (linear) | 86.63% | 89.47% | 83.33% | 90.00% | 86.29% | 0.924 | 86.53% ± 3.44% |

Figures: D1–D4

## E) Knowledge gained

- **Most common keywords:** APPLE, LIGHT-BODIED, PEAR, FRESH, PEACH, SPICE, MELON, MINERAL, ACIDITY, RIPE, LONG, WHITE, APRICOT, RICH, CREAM
- **Strongest 90+ keywords** (≥10 reviews): FULL-BODIED (100%), GRACE (100%), DENSE (100%), HARMONY (100%), HAZELNUT (100%), POWER (100%), DARK (100%), BEAUTY (100%), BLACK OLIVE (100%), GORGEOUS (100%), FINESSE (100%), LONG (98%), INTENSE (98%), CONCENTRATED (97%), ELEGANT (97%)
- **Strongest 90- keywords** (≥10 reviews): MEDIUM-BODIED (0%), STRAWBERRY (8%), DRY (19%), LIGHT-BODIED (22%), HERBS (24%), REFRESHING (26%), CRISP (30%), CLEAN (31%), AROMATIC (36%), FRESH (36%), ROUND (39%), FLORAL (39%), GREEN APPLE (39%), LIME (40%), EARTHY (41%)
- **90+ keywords agreed on by Naive Bayes and SVM:** APRICOT, BEAUTY, CONCENTRATED, DARK, ELEGANT, EXPRESSIVE, FINESSE, FULL-BODIED, HONEY, LONG, LONG FINISH, POWER
- Figures: E1–E3
