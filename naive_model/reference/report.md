# Baseline FFNN one-hot: confronto con ProbJanusDDG

Sequenze complete one-hot: FFNN condivisa 42→64→64 con ReLU sui residui (WT, WT−mutante, distanza relativa firmata e assoluta), pooling medio/massimo mascherato e vettore del sito mutato, readout 192→64→2. Nessun taglio delle sequenze. Nessun ESM, convoluzione o attention. Training direzionale con inversi e Gaussian NLL (beta=0); inferenza con media antisimmetrica e scala simmetrica positiva. Adam 1e-4, batch 6; GPU float32 (Janus usava GPU bfloat16).

Stessi dati e split verificati tramite hash e uguaglianza del piano. Per ogni fold esterno, tre fold per training interno, uno per MSE medio per proteina su 300 epoche; refit da zero sui quattro fold per l'epoca selezionata. Ogni fold esterno è escluso da training e selezione della propria epoca. Semi identici al protocollo Janus. Predizione principale: mediana dei cinque refit. Media riportata come riferimento.

Il CV+ usa la stessa funzione `weighted_bounds` di Janus, con pesi 1/(P*n_p), quantili empirici dei candidati e accoppiamento tra residui OOF e predizioni di test dello stesso refit. Standard e adattivo, livelli 50–95%. Nessun intervallo è ricentrato sulla mediana. La variante pesata resta empirica, senza una nuova garanzia esatta dimostrata. Coverage per mutazione e coverage media per proteina sono riportate separatamente.

Questo confronto cambia sia architettura sia rappresentazione con aggregazione globale senza interazioni esplicite tra residui: non isola il solo effetto degli embedding. L'architettura e gli iperparametri sono fissati prima della valutazione del baseline; i benchmark erano già stati esaminati per Janus. S461L si sovrappone a S669L; non sono repliche indipendenti. I confronti sono descrittivi, senza test di significatività.

## Epoche selezionate

| outer_fold | early_stopping_fold | selected_epoch | selected_mse_protein | minimum_at_cap |
|---|---|---|---|---|
| 0 | 1 | 204 | 3.0219 | False |
| 1 | 2 | 66 | 2.9031 | False |
| 2 | 3 | 153 | 3.3227 | False |
| 3 | 4 | 152 | 1.8667 | False |
| 4 | 0 | 34 | 2.4390 | False |

## Performance puntuale (mediana)

| model | dataset | aggregation | pearson | spearman | mse | rmse | mae | mse_equal_protein |
|---|---|---|---|---|---|---|---|---|
| ProbJanusDDG | S669L | median | 0.5296 | 0.5492 | 2.0322 | 1.4256 | 0.9925 | 3.0141 |
| ProbJanusDDG | S461L | median | 0.7037 | 0.6749 | 0.9723 | 0.9860 | 0.7363 | 1.7215 |
| One-hot FFNN | S669L | median | 0.3285 | 0.3135 | 2.7569 | 1.6604 | 1.2284 | 3.8750 |
| One-hot FFNN | S461L | median | 0.4302 | 0.3985 | 1.7441 | 1.3206 | 0.9956 | 2.8100 |

## Intervalli al 90%

| model | dataset | method | coverage | coverage_equal_protein | mean_width | interval_score |
|---|---|---|---|---|---|---|
| ProbJanusDDG | S669L | standard | 0.9163 | 0.8676 | 4.9639 | 6.7156 |
| ProbJanusDDG | S669L | adaptive | 0.9253 | 0.8687 | 5.0037 | 6.5507 |
| ProbJanusDDG | S461L | standard | 0.9761 | 0.9656 | 4.9652 | 5.1877 |
| ProbJanusDDG | S461L | adaptive | 0.9805 | 0.9385 | 4.9668 | 5.1092 |
| One-hot FFNN | S669L | standard | 0.9088 | 0.8631 | 5.6415 | 7.5479 |
| One-hot FFNN | S669L | adaptive | 0.9028 | 0.8641 | 5.8422 | 7.9223 |
| One-hot FFNN | S461L | standard | 0.9588 | 0.9159 | 5.6410 | 6.2172 |
| One-hot FFNN | S461L | adaptive | 0.9631 | 0.9200 | 5.9272 | 6.4185 |

## Segno: livelli 50%, 80%, 90%

| model | dataset | nominal | n_calls | call_percent | n_correct | sign_accuracy_percent |
|---|---|---|---|---|---|---|
| ProbJanusDDG | S669L | 0.5000 | 229 | 34.2302 | 217 | 94.7598 |
| ProbJanusDDG | S669L | 0.8000 | 48 | 7.1749 | 48 | 100.0000 |
| ProbJanusDDG | S669L | 0.9000 | 4 | 0.5979 | 4 | 100.0000 |
| ProbJanusDDG | S461L | 0.5000 | 190 | 41.2148 | 187 | 98.4211 |
| ProbJanusDDG | S461L | 0.8000 | 47 | 10.1952 | 47 | 100.0000 |
| ProbJanusDDG | S461L | 0.9000 | 5 | 1.0846 | 5 | 100.0000 |
| One-hot FFNN | S669L | 0.5000 | 189 | 28.2511 | 154 | 81.4815 |
| One-hot FFNN | S669L | 0.8000 | 13 | 1.9432 | 13 | 100.0000 |
| One-hot FFNN | S669L | 0.9000 | 0 | 0.0000 | 0 | nan |
| One-hot FFNN | S461L | 0.5000 | 141 | 30.5857 | 123 | 87.2340 |
| One-hot FFNN | S461L | 0.8000 | 11 | 2.3861 | 11 | 100.0000 |
| One-hot FFNN | S461L | 0.9000 | 0 | 0.0000 | 0 | nan |

## Figure

- `figures/S669L_comparison.pdf` e `S461L_comparison.pdf`: copertura, ampiezza, chiamate di segno e accuratezza.
- `figures/point_predictions.pdf`: osservato contro predetto.
- `figures/epoch_selection.pdf`: selezione delle epoche per fold.
