# Note per la presentazione

Slide in inglese; esposizione suggerita in italiano. Durata principale: circa 18 minuti. Le quattro slide di riserva non entrano nel tempo previsto.

## 1. Titolo

Tempo: 30 secondi. Presenta il problema: oltre a prevedere il cambiamento di stabilita, vogliamo capire quanto possiamo usare una predizione per prendere una decisione. Il contributo combina calibrazione conformal, peso uguale delle proteine e valutazione del segno.

## 2. Protein stability and mutation effects

Tempo: 1 minuto. Introduci il significato biologico della stabilita senza trasformare la slide in una lezione di termodinamica. Specifica subito la convenzione del segno del nostro paper. Non confondere stabilita con funzione o patogenicita: sono quantita diverse.

## 3. JanusDDG: sequence-based prediction

Tempo: 1 minuto. Presenta JanusDDG come base architetturale e collega la simmetria di inversione al problema fisico. La pubblicazione originale tratta anche mutazioni multiple; il presente studio valuta singole sostituzioni. Non dedicare tempo ai dettagli degli strati di attenzione. Fonte: https://doi.org/10.1038/s42003-026-09632-9.

## 4. Uncertainty for individual predictions

Tempo: 1 minuto. Spiega che una buona MAE non permette di conoscere l'errore della prossima mutazione. Riconosci i precedenti BayeStab e FoldX senza sostenere che siamo i primi a fornire intervalli. Il nostro obiettivo distingue la procedura di calibrazione e la popolazione bersaglio. Gli intervalli sono predittivi, non intervalli di confidenza della media. Fonti: Wang et al. (2022), doi:10.1002/pro.4467; Sapozhnikov et al. (2023), doi:10.1186/s12859-023-05537-0.

## 5. Conformal prediction: residual calibration

Tempo: 1 minuto e mezzo. Spiega i residui come errori osservati su dati non usati per addestrare il modello. Per il caso ordinario, q e la statistica d'ordine di indice ceil((m+1)(1-alpha)), con infinito se l'indice supera m. La garanzia richiede scambiabilita ed e marginale: non dice che ogni mutazione ha probabilita 90 percento di essere coperta. Il nostro metodo usa CV+ e dati raggruppati per proteina, quindi il risultato mostrato qui non si trasferisce automaticamente. Fonte: libro allegato, capitoli 1 e 3.

## 6. Equal calibration weight for each protein

Tempo: 1 minuto. I numeri A e B sono un esempio didattico, non un'analisi del dataset. Con pesi uniformi per mutazione la proteina A domina la calibrazione. Il nostro target evita che il numero di misure determini il peso della proteina. Questo non significa una distribuzione uniforme su tutte le possibili sequenze ne copertura condizionata garantita per ogni proteina.

## 7. ProbJanusDDG: mean and local scale

Tempo: 1 minuto e mezzo. Mostra i due output: media e scala positiva. L'inversione del verso cambia il segno della media e mantiene la scala. Il training usa una likelihood gaussiana, ma la calibrazione conformal non richiede residui gaussiani. Non presentare sigma come l'errore esatto della singola mutazione.

## 8. Protein-weighted CV+ calibration

Tempo: 1 minuto e mezzo. La scala normalizza il residuo della mutazione di calibrazione e lo riscalibra per la mutazione nuova. Ogni residuo rimane associato al modello che lo ha prodotto. La predizione puntuale e la mediana dei cinque modelli, ma l'intervallo non deriva semplicemente da questa mediana piu o meno un margine globale. Non confondere il livello nominale con il limite inferiore teorico della copertura.

## 9. Training and external evaluation

Tempo: 1 minuto. Presenta S669L come benchmark principale del talk. S2450 esclude anche le proteine oltre il 25 percento di identita con il benchmark S669. Descrivi brevemente la selezione interna delle epoche e il successivo refit: il fold esterno non entra ne nella selezione ne nel fitting. S461L e un sottoinsieme curato, non un secondo esperimento indipendente.

## 10. External point-prediction performance

Tempo: 1 minuto. Non presentare questo risultato come una vittoria sulla performance di JanusDDG o di tutti i competitor. Le prestazioni puntuali forniscono il contesto: l'obiettivo centrale del lavoro e integrare incertezza interpretabile. I numeri storici dei competitor provengono da benchmark pubblicati e non da un confronto rifatto in questo studio.

## 11. Mutation and protein coverage

Tempo: 1 minuto e mezzo. Questa e una slide centrale. La pesatura durante la calibrazione e diversa dalla pesatura della metrica di valutazione. La curva mostra copertura sulle mutazioni; la tabella esplicita il target con uguale peso delle proteine. Non dire che il 90 percento nominale e stato raggiunto sul target protein-balanced: qui osserviamo circa 87 percento. Le bande bootstrap sono puntuali e mantengono fissi i modelli.

## 12. Adaptive intervals redistribute width

Tempo: 1 minuto e mezzo. Parti dalla distribuzione delle ampiezze: la versione standard non ha ampiezza perfettamente costante perche CV+ combina modelli diversi. La versione adattiva assegna intervalli piu diversi alle mutazioni mantenendo una media simile. A destra, la differenza adattivo meno standard cresce mediamente dal quintile a basso errore a quello ad alto errore. I gruppi usano gli errori osservati solo dopo la valutazione, non per selezionare prospetticamente le mutazioni.

## 13. Learned scale and uncertainty groups

Tempo: 1 minuto. Il rango di incertezza e la media dei percentili di sigma rispetto al fold di calibrazione di ciascun modello. La correlazione 0.22 e moderata: sigma non misura precisamente l'errore individuale. La dispersione e la deviazione standard di popolazione delle cinque coperture, in punti percentuali. Il risultato descrive una riduzione della variabilita osservata, non una garanzia di copertura condizionata al gruppo.

## 14. Sign-call frequency and accuracy

Tempo: 2 minuti. Mostra il compromesso operativo tra numero di decisioni e accuratezza osservata. Non confondere un intervallo nominale del 50 percento con una probabilita del 50 percento che il segno sia corretto. Il 100 percento al 90 percento nominale deriva da sole quattro chiamate: non e evidenza di certezza. Le chiamate sono prevalentemente destabilizzanti e vanno interpretate nella composizione di questo benchmark. Le proteine raggiunte hanno almeno una mutazione chiamata, non tutte le mutazioni risolte.

## 15. Conclusions

Tempo: 1 minuto. Riporta il messaggio principale: la valutazione dell'incertezza affronta un problema complementare alla MAE. Chiudi mantenendo visibile il limite sul target protein-balanced. La teoria richiede ipotesi specifiche di campionamento e fitting; non offre una garanzia per ogni singola proteina o per l'accuratezza delle sole chiamate. Se c'e tempo o una domanda, usa le slide di riserva.

## 16. Backup: the curated S461L subset

Slide di riserva. S461L e piu conservativo: non raccontare questo come una seconda conferma indipendente del 90 percento. La composizione del benchmark cambia i risultati e l'adattivita non migliora sistematicamente ogni metrica, inclusa la media per proteina.

## 17. Backup: calibration of a one-hot baseline

Slide di riserva. Il confronto cambia sia architettura sia rappresentazione: non attribuire la differenza solo agli embedding. La calibrazione opera su entrambi i predittori; la qualita del modello determina quanto informativi possono essere gli intervalli. Le coperture per proteina restano inferiori al nominale in entrambi.

## 18. Backup: coverage assumptions

Slide di riserva. Da usare solo se l'appendice teorica e inclusa nella versione definitiva del paper. Il modello gerarchico assume proteine indipendenti, una legge delle mutazioni che non dipende dalla dimensione osservata del cluster dopo aver condizionato sulla proteina, e allineamento del test al target. I fold devono essere indipendenti dai dati per questo risultato. Queste ipotesi non sono automaticamente soddisfatte da fold basati sull'omologia. Il limite e finito-campione, debole, e non equivale alla copertura nominale del 90 percento.

## 19. Selected references

Slide di riserva per i riferimenti. Le figure e i valori di ProbJanusDDG provengono dal manoscritto allegato, versione prob_janus (6).pdf. Il libro e la versione pre-publication allegata del 6 marzo 2026.
