"""
beta-NLL (Seitzer et al., 2022), con beta = 0,5.

ARGOMENTO TEORICO. Nella verosimiglianza gaussiana standard il gradiente rispetto a mu e' scalato
da 1/sigma^2: i punti gia' predetti bene, dove sigma e' piccola, dominano l'ottimizzazione e la
spingono ancora piu' in basso. Il peso sigma^(2*beta), STACCATO dal grafo, annulla esattamente
quella scala per beta = 1 e la dimezza per beta = 0,5.

CHE COSA SUCCEDE IN PRATICA. Il correttivo non impedisce a sigma di restringersi: con beta = 0,5
attivo, la mediana di sigma cala da 1,104 a 0,068 e sd(z) sale da 1,3 a 18,2 lungo le prime 173
epoche (media sui cinque fold; il valore calibrato di sd(z) e' 1). Nello stesso intervallo
rho(sigma,|errore|) SALE da +0,039 a +0,208.

Scala e ordinamento sono grandezze indipendenti, e il protocollo usa solo la seconda: la larghezza
degli intervalli non viene mai da sigma ma dal quantile conforme, che la riscala. Sigma conta come
ordinamento, mai come larghezza.

Ne segue che beta NON e' una scelta portante: il collasso che dovrebbe prevenire avviene comunque e
non danneggia il risultato. 0,5 e' un default conservativo, non un ingrediente necessario.

La costante 0,5*log(2*pi) e' omessa: non dipende dai parametri.
"""
import torch


def beta_nll_each(y, mu, sigma, beta=0.5):
    """Come beta_nll ma SENZA la media: serve a pesare i campioni uno per uno."""
    l = torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2
    if beta > 0:
        l = l * sigma.detach() ** (2 * beta)
    return l


def beta_nll(y, mu, sigma, beta=0.5):
    return beta_nll_each(y, mu, sigma, beta).mean()
