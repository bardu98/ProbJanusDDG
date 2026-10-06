# Verifica di riproducibilità del manoscritto

Data: 6 ottobre 2026. Manoscritto verificato: `paper_overleaf/paper_cvplus.tex`. Il manoscritto non è stato modificato.

## Esito

I risultati del modello principale sono riproducibili dai cinque checkpoint disponibili localmente, con aggregazione mediante mediana e calibrazione pesata per proteina. Ho ricalcolato le predizioni, le statistiche, gli intervalli e le figure. La verifica non comprende un nuovo addestramento completo né una dimostrazione delle proprietà teoriche del metodo.

- Le 15 verifiche di inferenza (calibrazione fuori campione e due test set, per ciascuno dei cinque modelli) restituiscono medie e scale **identiche bit per bit** alle predizioni archiviate: 8.100 predizioni complessive.
- Le 1.500 metriche di selezione interna, ricalcolate dalle predizioni di validation archiviate, coincidono con le curve salvate entro 4,44 × 10⁻¹⁶. Sono confermate le epoche **57, 232, 126, 283, 105**.
- Tutte le **otto figure quantitative PDF** coincidono visivamente con quelle rigenerate: confronto delle immagini rasterizzate, nessun pixel differente.
- Dei **306 controlli numerici**, 304 coincidono; due valori stampati richiedono correzzione di arrotondamento. Sono state verificate le tabelle del modello principale, del naive, del confronto bibliografico e dell'esclusione dello zero, insieme alle statistiche quantitative principali del testo.

## Valori da correggere

| Posizione nel manoscritto | Stampato | Ricalcolato | Correzione |
|---|---:|---:|---:|
| Riga 955, Spearman S461L | 0,68 | 0,6748859336 | **0,67** |
| Riga 559, contrasto fra quintili estremi di errore, S669L | 0,39 | 0,3841003918 | **0,38**, oppure 0,384 |

Il secondo contrasto deriva dalla differenza fra 0,2787136303 e −0,1053867615. Le medie stampate come +0,28 e −0,11 sono corrette, ma sottrarre quei valori già arrotondati produce 0,39 anziché l'arrotondamento della differenza originale.

| Dataset | Pearson | Spearman | MAE | MSE |
|---|---:|---:|---:|---:|
| S669L | 0,529575 | 0,549218 | 0,992541 | 2,032247 |
| S461L | 0,703728 | 0,674886 | 0,736272 | 0,972265 |

Le coperture, le larghezze, gli interval score, le correlazioni fra incertezza ed errore e le percentuali di segni corretti risultano coerenti con il manoscritto. Ad esempio, a livello nominale 90%, la copertura adattiva sulle mutazioni è 92,5262% su S669L e 98,0477% su S461L; pesando le proteine ugualmente è 86,87% e 93,85%. Queste due definizioni di copertura non vanno scambiate.

## Elementi da sistemare per la distribuzione pubblica

1. **Modello naive.** I risultati numerici sono stati verificati dalle sue predizioni originali, conservate in `naive_reference/` con provenienza e hash. Il repository principale non contiene però la sua pipeline completa di training e i checkpoint. Per riprodurre anche questo esperimento da zero occorre distribuirli. Sono disponibili localmente in `../cv+_21sett_2026/naive_regressor/`; la piccola copia per questo audit non sostituisce quella pipeline.
2. **Asset del modello principale.** Embedding e checkpoint sono disponibili localmente e verificati, ma esclusi dal normale contenuto Git. Prima della pubblicazione occorre rendere disponibile un archivio persistente e inserire il relativo collegamento e gli hash. La presenza dei file nella macchina non equivale alla disponibilità per un lettore del paper.
3. **Abstract separato.** `paper_overleaf/abstract.tex` contiene una descrizione precedente del metodo e risultati differenti. Non è incluso dal manoscritto principale, quindi non altera il PDF corrente; va aggiornato o rimosso prima di usarlo per la sottomissione.
4. **Ambiente e batching.** L'inferenza verificata usa il modello storico, BF16 su CUDA, batch di 32 elementi ordinati per lunghezza e le versioni registrate nei log. L'architettura conserva il comportamento dipendente dal padding del batch: cambiare batching può cambiare le predizioni. La documentazione del repository descrive questa condizione; occorre mantenerla esplicita per la riproduzione esatta.

## Provenienza dei dataset e dei confronti

Le tabelle di confronto bibliografico sono coerenti con i quattro fogli originali della [repository JanusDDG](https://github.com/compbiomed-unito/JanusDDG/tree/main/Paper_Figures), scaricati al commit `7ef53fde72c9b9b715b0a9f2f80910ebf927bd59`. URL e hash sono in `external_sources/manifest.json`. Questa verifica controlla la trascrizione delle prestazioni pubblicate, non riesegue i predittori concorrenti.

La soglia del 25% per la costruzione di S2450 e per gli split è descritta nel [paper originale DDGemb](https://academic.oup.com/bioinformatics/article/41/1/btaf019/7952013). Quindi non è da correggere sulla sola base della pagina web del dataset, che riporta una diversa soglia del 30% e copertura del 40%. Resta opportuno chiarire l'unità di conteggio delle proteine nel paragrafo dei dati: 131 proteine iniziali meno 18 sequenze escluse non dà 115; la rimappatura su sequenze complete può cambiare le unità contate. Sono confermate direttamente nei dati utilizzati le **115 sequenze WT** di S2450, **87** di S669L e **46** di S461L. Non ho rieseguito gli allineamenti per certificare autonomamente ogni soglia di identità.

La frase sulle coperture FoldX fino al 79% è confermata dalla tabella 5 del [lavoro citato](https://doi.org/10.1186/s12859-023-05537-0): non è un errore di trascrizione.

S461L e S669L non sono test indipendenti: condividono 45 sequenze WT e 460 coppie di mutazione. Il conteggio serve a interpretare l'evidenza sperimentale, senza cambiare i risultati riportati.

## Ripetere la verifica

Dalla cartella `git_ProbJanusDDG/`:

```bash
python scripts/audit_paper.py
python scripts/verify_checkpoint_predictions.py
```

Il secondo comando richiede gli asset e l'ambiente GPU descritti nella documentazione. Per verificare anche le 1.500 curve dalle predizioni interne originali, disponibili nella cartella locale dell'esperimento:

```bash
python scripts/audit_paper.py --inner-predictions-dir '../cv+_21sett_2026/selection'
```

Le evidenze sono in `results.json`, `numerical_checks.csv`, `epoch_checks.csv`, `figure_checks.csv`, `recomputed_sign_calls.csv` e `checkpoint_predictions/verification.json`. Gli script verificano i risultati senza modificare il manoscritto e senza effettuare selezione o addestramento sui test set.

## Aggiornamento: pipeline naive inclusa

La pipeline completa è ora distribuita in `../naive_model/`, con architettura invariata, cinque checkpoint finali, predizioni per fold e curve di selezione. Il punto 1 sopra descrive lo stato precedente a questa integrazione; non è più un elemento mancante. I percorsi e il launcher sono stati resi portabili; le nuove esecuzioni sono separate in `runs/naive/`. La verifica operativa è in `naive_model/reference/verification.json`.
