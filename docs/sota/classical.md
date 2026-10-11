# État de l'art : complétion de matrice classique (non profonde) et informations annexes

**Contexte visé** : copie de MovieLens ml-latest-small, matrice de 610 utilisateurs × 4980 films, notes de 0,5 à 5 par pas de 0,5. Le fit se fait sur train+test (≈ 63 k notes, densité ≈ 2,1 %). Le score est le RMSE sur un jeu d'évaluation caché, avec en plus l'exactitude des notes arrondies et le temps d'exécution. Contraintes : CPU seulement, 4 threads, 12 Go, 1200 s. La seule information annexe est `namesngenre.npy` (titres « Toy Story (1995) » et genres séparés par `|`).

**Résultats locaux actuels** (fit sur train, évaluation sur test) : biais 0,875 ; soft-impute 0,8715 ; MF biaisée SGD/ALS ≈ 0,88 ; item-kNN 0,890. Meilleur score du classement de la plateforme : ≈ 0,80-0,815.

> **Légende** : les chiffres sourcés viennent des articles ou pages citées. Les « gains attendus » sur notre jeu sont des **estimations** tirées de la littérature et de l'expérience sur ML-100K et ml-latest-small. Ils restent à valider en local.

---

## 0. Remarques préliminaires importantes pour notre cadre

1. **L'écart local/plateforme est en partie mécanique.** En local, on fit sur train seul. Sur la plateforme, on fit sur train+test, soit plus de données par utilisateur. Sur une matrice aussi creuse (≈ 100 notes/utilisateur en moyenne, médiane bien plus basse), le RMSE baisse sensiblement quand les données augmentent. Un 0,87 local vaut donc probablement ≈ 0,85-0,86 sur la plateforme. Pour reproduire le régime de la plateforme, il faut faire de la validation croisée K-fold sur train+test, ce qui est plus représentatif qu'un seul split train → test.
2. **Les méthodes « à moyenne de nombreux modèles » dominent sur petites données.** Les plus efficaces sont les méthodes bayésiennes par MCMC (BPMF, Bayesian FM) et le blending. Un modèle point-estimate unique bien régularisé plafonne vite, et nos 0,87-0,88 sont typiques de ce plafond.
3. **Le feedback implicite est partiel ici.** Sur Netflix, les positions (u, i) du jeu *qualifying* étaient connues, ce qui donnait à SVD++ un signal « quels films l'utilisateur a notés » incluant les cibles. Ici, seules les positions train+test sont connues, et les positions eval sont cachées. On peut donc exploiter N(u) = films notés dans train+test, mais pas les positions cibles. Le gain de SVD++ et du « flipped » de Rendle sera donc plus faible que dans la littérature.
4. **Pas de timestamps.** timeSVD++ et tous les biais temporels sont inapplicables. Seule l'année de sortie (extraite du titre) est un substitut faible, utilisable comme feature d'item.

---

## 1. Famille Koren (Netflix Prize)

### 1.1 Prédicteurs de base avec shrinkage
- **Koren, Y. (2010).** *Factor in the Neighbors: Scalable and Accurate Collaborative Filtering.* ACM TKDD 4(1). https://doi.org/10.1145/1644873.1644874 (version KDD 2008 ci-dessous).
- Modèle : `b_ui = μ + b_u + b_i`.
- Estimation découplée régularisée, à faire en premier les items puis les utilisateurs :
  - `b_i = Σ_{u∈R(i)} (r_ui − μ) / (λ2 + |R(i)|)`, avec λ2 ≈ 25 ;
  - `b_u = Σ_{i∈R(u)} (r_ui − μ − b_i) / (λ3 + |R(u)|)`, avec λ3 ≈ 10.
- On peut aussi les estimer par moindres carrés régularisés : `min Σ (r_ui − μ − b_u − b_i)² + λ(Σb_u² + Σb_i²)`, résolus par ALS en quelques itérations. C'est l'option `BaselineOnly` de Surprise, avec par défaut `reg_i=10, reg_u=15, n_epochs=10`.
- Notre 0,875 correspond à ce modèle. Avec λ bien réglé par CV, on peut viser ≈ 0,865-0,87 en local.

### 1.2 SVD++ et modèle intégré
- **Koren, Y. (2008).** *Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model.* KDD 2008, pp. 426-434. https://doi.org/10.1145/1401890.1401944
- Équation de SVD++ :
  `r̂_ui = μ + b_u + b_i + q_iᵀ ( p_u + |N(u)|^{-1/2} Σ_{j∈N(u)} y_j )`,
  où N(u) est l'ensemble des items pour lesquels u a fourni un feedback implicite (ici, les films notés dans train+test).
- SGD : `e_ui = r_ui − r̂_ui`.
  - `b_u += γ1 (e − λ6 b_u)` ;
  - `q_i += γ2 (e·(p_u + |N|^{-½}Σy_j) − λ7 q_i)` ;
  - `p_u += γ2 (e q_i − λ7 p_u)` ;
  - `∀j∈N(u) : y_j += γ2 (e |N|^{-½} q_i − λ7 y_j)`.
  - Valeurs de l'article : γ1 = γ2 = 0,007, λ6 = 0,005, λ7 = 0,015, décroissance du pas ×0,9 par epoch, environ 30 epochs.
  - Valeurs par défaut de Surprise `SVDpp` : `n_factors=20, n_epochs=20, lr_all=0.007, reg_all=0.02`.
- **Modèle intégré** : SVD++ plus un terme de voisinage global appris :
  `+ |R^k(i;u)|^{-½} Σ_{j∈R^k(i;u)} (r_uj − b_uj) w_ij + |N^k(i;u)|^{-½} Σ_{j∈N^k(i;u)} c_ij`.
  Les w_ij et c_ij sont appris par SGD, et R^k se restreint aux k plus proches voisins de i (k ≈ 300 sur Netflix).
- Ordres de grandeur Netflix (probe, de mémoire, à vérifier dans l'article) : SVD f=50 ≈ 0,905 ; SVD++ f=50 ≈ 0,895 ; intégré ≈ 0,887. Le gain de SVD++ sur SVD est d'environ 1 %, et l'intégré ajoute encore ≈ 0,7 %.
- Sur Surprise, ML-1M : SVD 0,873 contre SVD++ 0,862, soit −0,011 (voir §9).
- **Coût** : SVD++ naïf est O(|N(u)|·f) par note. En Python pur c'est trop lent. Surprise en Cython prend ≈ 1,5 min sur ML-100K, et la version vectorisée par utilisateur (mise à jour des y_j une fois par utilisateur) est bien plus rapide. Avec 63 k notes, une implémentation numba ou numpy « par utilisateur » prend quelques secondes.

### 1.3 timeSVD++
- **Koren, Y. (2009).** *Collaborative Filtering with Temporal Dynamics.* KDD 2009 ; version longue dans CACM 53(4), 2010. https://doi.org/10.1145/1557019.1557072
- Il ajoute des biais `b_u(t)`, `b_i(Bin(t))`, `p_u(t)`.
- **Inapplicable chez nous, faute de timestamps.** L'année de sortie du film n'est pas un temps de notation. On peut seulement l'utiliser comme feature d'item, par exemple un biais d'item par tranche d'année.

### 1.4 Article de synthèse
- **Koren, Y., Bell, R., Volinsky, C. (2009).** *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer 42(8):30-37. https://doi.org/10.1109/MC.2009.263
- Objectif : `min Σ (r_ui − μ − b_u − b_i − q_iᵀp_u)² + λ(‖q_i‖² + ‖p_u‖² + b_u² + b_i²)`, résolu par SGD ou ALS.
- L'article présente aussi l'ajout de sources implicites et d'attributs utilisateurs : `p_u + |N(u)|^{-½}Σx_i + Σ_{a∈A(u)} y_a`.
- Ce terme d'attributs se transpose côté item avec nos genres : `q_i + Σ_{g∈G(i)} z_g`, ce qui ancre le facteur d'un film peu noté sur ses genres.

### 1.5 Rendle, Zhang, Koren (2019) : référence sur les baselines
- **Rendle, S., Zhang, L., Koren, Y. (2019).** *On the Difficulty of Evaluating Baselines: A Study on Recommender Systems.* arXiv:1905.01395. https://arxiv.org/abs/1905.01395
- Sur ML-10M (split 90:10), des baselines classiques bien réglées battent tous les modèles « nouveaux » publiés :

| Méthode | RMSE ML-10M |
|---|---|
| Biased MF (valeur souvent rapportée) | 0,803 |
| BPMF (valeur souvent rapportée) | 0,8197 |
| LLORMA / AutoRec / MRMA (meilleur publié) | 0,7815 / 0,782 / 0,7634 |
| **SGD-MF bien réglée** (512 dims, 128 epochs, λ=0,04, γ=0,003) | **0,7720** |
| **Bayesian MF** (Gibbs, 512 dims) | **0,7633** |
| Bayesian timeSVD | 0,7587 |
| **Bayesian SVD++** | **0,7563** |
| Bayesian timeSVD++ | 0,7523 |
| Bayesian timeSVD++ « flipped » | 0,7485 |

- Le « flipped » ajoute davantage de feedback implicite (côté item, de façon symétrique). Les modèles bayésiens sont implémentés comme des FM par MCMC (libFM).
- Leçons : (i) le MCMC bat le SGD d'environ 0,01 ; (ii) SVD++ (implicite) apporte encore environ 0,007 ; (iii) **la dimension peut être grande en bayésien sans sur-apprentissage** ; (iv) régler soigneusement λ, l'initialisation (σ = 0,1) et le nombre d'epochs.

---

## 2. Approches bayésiennes

### 2.1 PMF
- **Salakhutdinov, R., Mnih, A. (2007).** *Probabilistic Matrix Factorization.* NIPS 20. https://papers.nips.cc/paper/2007/hash/d7322ed717dedf1eb4e6e52a37ea7bcd-Abstract.html
- Modèle : `r_ij ~ N(u_iᵀv_j, σ²)`, avec `u_i ~ N(0, σ_U² I)` et `v_j ~ N(0, σ_V² I)`. Le MAP équivaut à la MF régularisée avec `λ_U = σ²/σ_U²`.
- Les auteurs proposent aussi un PMF contraint, où le facteur utilisateur est décalé par la moyenne des W des items notés. C'est l'ancêtre de SVD++ et c'est utile pour les utilisateurs peu actifs.
- Une sortie logistique `g(·)` ramène les notes dans [0, 1].

### 2.2 BPMF (Gibbs)
- **Salakhutdinov, R., Mnih, A. (2008).** *Bayesian Probabilistic Matrix Factorization using Markov Chain Monte Carlo.* ICML 2008. https://doi.org/10.1145/1390156.1390267
- Modèle : `U_i ~ N(μ_U, Λ_U^{-1})`, `V_j ~ N(μ_V, Λ_V^{-1})`, avec un hyperprior Normal-Wishart sur (μ, Λ) (`β0=2, ν0=D, W0=I, μ0=0`) et un bruit de précision α fixé (α = 2).
- Gibbs alterné : chaque `U_i | V, R` est gaussien, de précision `Λ_U + α Σ_j V_j V_jᵀ`. Puis on tire les hyperparamètres.
- Prédiction : `r̂ = (1/T) Σ_t U_i^(t)ᵀ V_j^(t)`, la moyenne de T échantillons.
- Sur Netflix, BPMF bat nettement PMF/SVD réglés (≈ 1-2 % de RMSE). Le gain grandit avec D, puisque la moyenne bayésienne empêche le sur-apprentissage. Il est **particulièrement fort pour les utilisateurs avec peu de notes**, ce qui est exactement notre régime.
- Coût : O(|R|·D² + (N+M)·D³) par itération. Pour 63 k notes, D = 10-30, 610 + 4980 vecteurs et 200-500 itérations, cela prend quelques dizaines de secondes en numpy vectorisé.
- Pour un ML-10M bien réglé, voir le Bayesian MF de Rendle 2019 (0,7633).

### 2.3 Bayes variationnel
- **Lim, Y. J., Teh, Y. W. (2007).** *Variational Bayesian Approach to Movie Rating Prediction.* KDD Cup & Workshop 2007. https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/variational-Lim.pdf
- **Nakajima, S., Sugiyama, M. (2011).** *Theoretical Analysis of Bayesian Matrix Factorization.* JMLR 12. (VBMF, sélection automatique du rang.)
- Le VB est plus rapide que le MCMC mais un peu moins bon. Sur Netflix, le VB était comparable à la MF régularisée bien réglée et utile en blending.

### 2.4 Bayesian Factorization Machines (MCMC FM)
Voir §5. C'est le moyen le plus simple d'avoir BPMF + SVD++ + genres + année dans **un seul** modèle MCMC (package `myfm`).

---

## 3. Approches convexes / norme nucléaire

### 3.1 Soft-Impute
- **Mazumder, R., Hastie, T., Tibshirani, R. (2010).** *Spectral Regularization Algorithms for Learning Large Incomplete Matrices.* JMLR 11:2287-2322. https://jmlr.org/papers/v11/mazumder10a.html
- Problème : `min_Z ½‖P_Ω(X − Z)‖_F² + λ‖Z‖_*`.
- Itération : `Z ← S_λ( P_Ω(X) + P_Ω⊥(Z) )`, où `S_λ(W) = U diag((σ_i − λ)_+) Vᵀ` est le seuillage doux des valeurs singulières.
- On parcourt un chemin de λ décroissant avec warm-start, et on choisit λ par CV.
- Notre 0,8715. Points clés : **centrer d'abord par les biais** (bi-scale, cf. `softImpute::biScale`). On peut aussi ajouter un « debiasing » final, c'est-à-dire réajuster les valeurs singulières par moindres carrés sur Ω.

### 3.2 softImpute-ALS
- **Hastie, T., Mazumder, R., Lee, J. D., Zadeh, R. (2015).** *Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares.* JMLR 16:3367-3402. https://jmlr.org/papers/v16/hastie15a.html
- On résout `min ½‖P_Ω(X − ABᵀ)‖² + λ/2 (‖A‖² + ‖B‖²)` en ALS sur la matrice « remplie ». C'est équivalent à la norme nucléaire si le rang r est assez grand.
- C'est beaucoup plus rapide que Soft-Impute basé SVD. Implémenté dans `fancyimpute` et dans le package R `softImpute`.

### 3.3 SVT
- **Cai, J.-F., Candès, E. J., Shen, Z. (2010).** *A Singular Value Thresholding Algorithm for Matrix Completion.* SIAM J. Optimization 20(4):1956-1982. https://doi.org/10.1137/080738970
- Itération : `Z^k = S_τ(Y^{k−1})`, puis `Y^k = Y^{k−1} + δ P_Ω(M − Z^k)`.
- Valeurs typiques : τ ≈ 5·√(nm), δ ≈ 1,2·nm/|Ω|.
- Pensé pour une reconstruction exacte (bas rang, sans bruit). Sur des notes bruitées, il fait moins bien que Soft-Impute et n'apporte rien de plus.

### 3.4 Norme de trace pondérée (échantillonnage non uniforme)
- **Salakhutdinov, R., Srebro, N. (2010).** *Collaborative Filtering in a Non-Uniform World: Learning with the Weighted Trace Norm.* NIPS 23. https://papers.nips.cc/paper/2010/hash/0f1b4c1b0c7ab5a8c2e8c8e5e0a1f6b8-Abstract.html (titre et auteurs certains ; le hash de l'URL est à vérifier).
- Norme : `‖X‖_tr(p,q) = ‖diag(√p) X diag(√q)‖_tr`, où p et q sont les marginales empiriques d'échantillonnage des lignes et colonnes.
- Forme factorisée équivalente : **régulariser chaque facteur proportionnellement à sa fréquence**, soit `λ Σ_u |R(u)| ‖p_u‖² + λ Σ_i |R(i)| ‖q_i‖²`. C'est exactement le « λ par note » de la SGD classique.
- Sur Netflix, la version pondérée améliore nettement la trace-norm non pondérée.
- Variante lissée : **Foygel, Salakhutdinov, Shamir, Srebro (2011).** *Learning with the Weighted Trace-norm under Arbitrary Sampling Distributions.* NIPS 24. Elle mélange p avec l'uniforme : `p̃ = (1−ζ)p + ζ/n`.
- **Conséquence pratique** : le Soft-Impute standard (non pondéré) sur-régularise les films populaires et sous-régularise les rares. Passer à un ALS avec `λ·(n_u)^α` et `λ·(n_i)^α`, en réglant α ∈ [0, 1], est une amélioration peu coûteuse.

### 3.5 Max-norm
- **Srebro, N., Rennie, J., Jaakkola, T. (2005).** *Maximum-Margin Matrix Factorization.* NIPS 17. https://papers.nips.cc/paper/2004/hash/e0688d13958a19e087e123148555e4b4-Abstract.html
- **Lee, J., Recht, B., Salakhutdinov, R., Srebro, N., Tropp, J. (2010).** *Practical Large-Scale Optimization for Max-Norm Regularization.* NIPS 23.
- Contrainte : `‖X‖_max = min_{X=UVᵀ} max_i‖U_i‖ · max_j‖V_j‖ ≤ B`, implémentée par SGD projetée (on renormalise chaque ligne à norme ≤ √B).
- Robuste à l'échantillonnage non uniforme, et comparable à la trace-norm pondérée sur Netflix. Intérêt marginal pour nous par rapport au 3.4.

---

## 4. Voisinages raffinés

### 4.1 KNNBaseline (Surprise) et similarités Pearson-baseline rétrécies
- **Koren 2010 (TKDD)**, §2.2 ; **Bell, R., Koren, Y. (2007).** *Scalable Collaborative Filtering with Jointly Derived Neighborhood Interpolation Weights.* ICDM 2007. https://doi.org/10.1109/ICDM.2007.90
- Prédiction : `r̂_ui = b_ui + Σ_{j∈N^k_i(u)} s_ij (r_uj − b_uj) / Σ_j |s_ij|`.
- Similarité Pearson-baseline rétrécie : on calcule une corrélation sur les résidus `r − b` (et non sur les notes brutes), puis on la rétrécit :
  `ŝ_ij = (n_ij − 1)/(n_ij − 1 + λ) · ρ_ij`, avec λ = 100 par défaut dans Surprise (`shrinkage=100`).
- Configuration Surprise recommandée : `KNNBaseline(k=40, sim_options={'name':'pearson_baseline','user_based':False,'shrinkage':100})`, avec `bsl_options={'method':'als','reg_i':10,'reg_u':15}`.
- Notre item-kNN à 0,890 vient très probablement de l'absence de baseline et de shrinkage. **KNNBaseline item-based est attendu ≈ 0,86-0,865 en local**, avec une erreur peu corrélée à celle de la MF, ce qui en fait un excellent membre de blend.
- Raffinement Koren 2010 : poids d'interpolation **appris** (global neighborhood model). Il est intégré au modèle intégré du §1.2.

### 4.2 SLIM
- **Ning, X., Karypis, G. (2011).** *SLIM: Sparse Linear Methods for Top-N Recommender Systems.* ICDM 2011. https://doi.org/10.1109/ICDM.2011.134
- Problème : `min_W ½‖A − AW‖² + β/2 ‖W‖² + λ‖W‖_1`, avec `W ≥ 0` et `diag(W) = 0`, résolu par ElasticNet colonne par colonne.
- Conçu pour le **top-N implicite**, pas pour le RMSE. Une version « résidus » (A = matrice des résidus après biais, sans contrainte de positivité) revient à une régression item-item apprise, proche du global neighborhood de Koren.
- Avec 4980 items, cela fait 4980 ElasticNet : faisable (≈ 1-3 min), mais gain incertain. Priorité basse.
- Cousin récent : **EASE** (Steck, WWW 2019), en forme close `B = I − P·diag(1/diag(P))` avec `P = (XᵀX + λI)^{-1}`. Il porte aussi sur l'implicite, avec une inversion 4980² ≈ 2 s.

---

## 5. Factorization Machines

- **Rendle, S. (2010).** *Factorization Machines.* ICDM 2010, pp. 995-1000. https://doi.org/10.1109/ICDM.2010.127
- **Rendle, S. (2012).** *Factorization Machines with libFM.* ACM TIST 3(3):57. https://doi.org/10.1145/2168752.2168771
- **Freudenthaler, C., Schmidt-Thieme, L., Rendle, S. (2011).** *Bayesian Factorization Machines.* NIPS Workshop on Sparse Representation and Low-rank Approximation.
- Modèle : `ŷ(x) = w0 + Σ_k w_k x_k + Σ_{k<l} ⟨v_k, v_l⟩ x_k x_l`, calculé en O(kn) via `½ Σ_f [(Σ_k v_kf x_k)² − Σ_k v_kf² x_k²]`.
- Encodage pour chaque note (u, i) :
  - one-hot utilisateur ;
  - one-hot film ;
  - **implicite utilisateur** : films notés par u dans train+test, normalisés en `1/√|N(u)|`. C'est l'équivalent de SVD++.
  - **implicite item « flipped »** : utilisateurs ayant noté i, normalisés en `1/√|N(i)|`.
  - **genres** du film en multi-hot, normalisés en `1/√|G(i)|` ;
  - **année** de sortie, en one-hot par tranche (par exemple par décennie ou par 5 ans) ;
  - éventuellement des mots du titre.
- En MCMC (libFM `-method mcmc`, ou `myfm` en Python), λ et μ ont des hyperpriors Gamma/Normal et sont échantillonnés. **Il n'y a donc presque pas d'hyperparamètres à régler** : seulement le rang k, le nombre d'itérations et l'écart type d'initialisation (≈ 0,1). Il faut le **grouping** des features (un hyperprior par groupe).
- Paquet **`myfm`** (Ohtsuki) : `pip install myfm`, wheels binaires, C++/Eigen multi-thread. https://github.com/tohtsky/myFM
  - Il gère le **format relationnel** (blocs implicites partagés, donc efficace pour SVD++), le grouping et un **ordered probit** (régression ordinale, utile pour l'exactitude, voir §7).
  - README ML-100K (fold u3) : FM basique (user + item, rank 8, 100 itérations) RMSE 0,903 ; avec grouping 0,896.
  - Les exemples du dépôt avec implicite (SVD++) et features utilisateur/film donnent typiquement environ −0,01 à −0,02 supplémentaires (estimation, à vérifier sur notre split).
- Rendle 2019 utilise justement ces FM MCMC pour ses « Bayesian SVD++ / timeSVD++ flipped ».
- **Coût chez nous** : avec 63 k lignes, environ 30 features actives par ligne plus les blocs implicites, rank 16-32 et 300-500 itérations, il faut compter 10-60 s sur 4 threads.

---

## 6. Hybrides et informations annexes

### 6.1 Collective Matrix Factorization
- **Singh, A. P., Gordon, G. J. (2008).** *Relational Learning via Collective Matrix Factorization.* KDD 2008. https://doi.org/10.1145/1401890.1401969
- On factorise conjointement `R ≈ U Vᵀ` et `G ≈ V Wᵀ` (G = matrice films × genres), avec V partagé et un poids α entre les deux pertes.
- C'est équivalent en esprit à `q_i + Σ_g z_g` (§1.4) ou à la FM avec genres.
- Package Python `cmfrec` (D. Cortes), avec un « offsets model » et le support des attributs d'items, en ALS rapide C/OpenMP. https://github.com/david-cortes/cmfrec

### 6.2 Biais et priors ancrés sur les genres et l'année
- Modèle hiérarchique simple et très efficace pour les films à **peu de notes**, qui sont majoritaires dans ml-latest-small (≈ la moitié des films ont ≤ 3 notes).
  `b_i = b_genre(i) + b_year(i) + δ_i`, avec `δ_i` rétréci vers 0 : `δ_i = Σ(r − μ − b_u − b_G − b_Y)/(λ + n_i)`.
- Le film peu noté hérite ainsi du biais moyen de ses genres et de son époque. Par exemple, les vieux films et les « Film-Noir » ou « Documentary » sont mieux notés, l'« Horror » moins bien.
- On peut faire la même chose sur les facteurs (prior de BPMF centré sur `Σ_g z_g`, ce que fait la FM MCMC automatiquement).
- **Gain attendu** : −0,003 à −0,01 sur la baseline. L'effet est surtout visible sur les items froids, mais il est vite absorbé par les modèles plus forts.
- Implémentation : 20 lignes numpy, < 1 s.

### 6.3 LightFM
- **Kula, M. (2015).** *Metadata Embeddings for User and Item Cold-start Recommendations.* CBRecSys @ RecSys 2015. arXiv:1507.08439. https://arxiv.org/abs/1507.08439
- Chaque item est la somme des embeddings de ses features (id + genres + mots du titre).
- Les pertes proposées sont logistic, BPR et WARP, toutes orientées **classement implicite**. Il n'y a pas de perte de régression RMSE. **Peu adapté** à notre métrique, et la FM (§5) couvre le même modèle en régression.

### 6.4 Embeddings textuels des titres
- Les titres contiennent : le nom, l'année, parfois un article déplacé (« Matrix, The (1999) ») et des suites (« Toy Story 2 »).
- Options sans téléchargement :
  - **année** par regex `\((\d{4})\)` ;
  - **TF-IDF** sur les mots et bigrammes de caractères (scikit-learn), puis SVD tronquée (k = 16-32) en features continues pour la FM ;
  - une **similarité titre** (franchises) comme prior de kNN.
- **sentence-transformers** (par exemple `all-MiniLM-L6-v2`, 384 dims, ≈ 90 Mo) :
  - il faut télécharger le modèle depuis HuggingFace au premier appel, et torch (≈ 200 Mo CPU) au pip install ;
  - sur la plateforme, l'accès réseau à l'exécution est **incertain** et le temps compte (1200 s), donc c'est risqué ;
  - **alternative sûre** : pré-calculer offline les embeddings des 4980 titres (4980×384 float16 ≈ 3,8 Mo) et les **commiter dans le dépôt**. C'est faisable si le règlement autorise des fichiers annexes et si les films sont identifiés de façon stable (ordre de `namesngenre.npy`).
  - Le gain attendu reste faible : la sémantique du titre aide surtout le cold-start, et les genres + l'année en capturent déjà l'essentiel. Estimation : < 0,005 de RMSE.
  - Une version plus riche « LLM knowledge » (l'embedding sait que « Godfather » est un chef-d'œuvre) pourrait donner un prior de qualité pour les films rares, mais cela frise l'information externe. À vérifier dans le règlement.

### 6.5 KMF et complétion inductive (IMC)
- **Zhou, T., Shan, H., Banerjee, A., Sapiro, G. (2012).** *Kernelized Probabilistic Matrix Factorization: Exploiting Graphs and Side Information.* SDM 2012. https://doi.org/10.1137/1.9781611972825.35
  - C'est un prior GP sur les colonnes de U et V, avec un noyau construit sur la side info (par exemple un noyau de genres/année, ou un Laplacien de graphe).
- **Jain, P., Dhillon, I. S. (2013).** *Provable Inductive Matrix Completion.* arXiv:1306.0626. https://arxiv.org/abs/1306.0626
- **Xu, M., Jin, R., Zhou, Z.-H. (2013).** *Speedup Matrix Completion with Side Information: Application to Multi-Label Learning.* NIPS 2013.
  - Le modèle IMC est `R ≈ X W Hᵀ Yᵀ`, avec X et Y les features utilisateurs et items.
  - Nous n'avons **aucune feature utilisateur** (X = I) et des features items pauvres (≈ 20 genres + année). L'IMC pur serait donc trop restrictif.
  - La version utile est la « dirty IMC » (Chiang, Hsieh, Dhillon, NIPS 2015, *Matrix Completion with Noisy Side Information*) : `R ≈ X W Yᵀ + N` avec N bas rang libre. Cela revient à MF + facteurs de genres, c'est-à-dire, encore une fois, ce que fait la FM.

---

## 7. Ensembling / blending et arrondis

### 7.1 Blending (Netflix Prize)
- **Koren, Y. (2009).** *The BellKor Solution to the Netflix Grand Prize.* https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf
- **Töscher, A., Jahrer, M., Bell, R. (2009).** *The BigChaos Solution to the Netflix Grand Prize.*
- **Piotte, M., Chabbert, M. (2009).** *The Pragmatic Theory Solution to the Netflix Grand Prize.*
- **Jahrer, M., Töscher, A., Legenstein, R. (2010).** *Combining Predictions for Accurate Recommender Systems.* KDD 2010. https://doi.org/10.1145/1835804.1835893
- **Bell, R., Koren, Y. (2007).** *Lessons from the Netflix Prize Challenge.* SIGKDD Explorations 9(2).
- Méthode : on entraîne M modèles sur train\holdout, puis on prédit le holdout et on fit une **régression ridge linéaire** `r ≈ Σ_m α_m r̂_m + β`. Ensuite on réentraîne chaque modèle sur tout et on applique les α.
- Raffinements : ajouter en entrée du blend des « méta-features » (log n_u, log n_i), un petit GBDT ou un MLP. C'est le blending non linéaire de BigChaos.
- Gains typiques : un blend de 3 à 5 modèles hétérogènes (MF bayésienne, SVD++, KNNBaseline, biais + genres) gagne **0,005 à 0,015** de RMSE sur le meilleur modèle seul.
- **Variante « K-fold stacking »** : les prédictions out-of-fold sur train+test servent à fitter les poids, ce qui évite de sacrifier un holdout, crucial vu la petite taille. Il faut tenir le budget : K = 5 × M modèles doit rester ≪ 1200 s.

### 7.2 Arrondis et compromis RMSE / exactitude
- Distribution des notes d'ml-latest-small (100 836 notes) : 4,0 (26,6 %) > 3,0 (19,9 %) > 5,0 (13,1 %) ≈ 3,5 (13,0 %) > 4,5 (8,5 %) > 2,0 (7,5 %) > 2,5 (5,5 %) > 1,0 (2,8 %) > 1,5 (1,8 %) > 0,5 (1,4 %).
- Les **notes entières sont sur-représentées** : 4 / 3 / 5 / 2 / 1 regroupent ≈ 70 %.
- **RMSE** : l'optimum est l'espérance conditionnelle, donc des prédictions continues, clippées dans [0,5 ; 5].
- **Exactitude** : l'optimum est le **mode** conditionnel. Arrondir la moyenne au 0,5 le plus proche n'est pas optimal :
  1. il faut **des seuils appris** sur un holdout, qui favoriseront les entiers. Par exemple, une prédiction 3,6 donne plutôt « 4,0 » que « 3,5 ». On optimise des seuils monotones entre classes par recherche directe (coordonnées) pour maximiser l'exactitude out-of-fold ;
  2. on peut aussi utiliser un **modèle ordinal** qui donne P(r = k), comme l'ordered probit de `myfm` ou une régression logistique ordinale sur les prédictions du blend, puis prendre l'argmax.
- **Si la même sortie est scorée pour les deux métriques**, arrondir coûte en RMSE. L'erreur de quantification uniforme sur ±0,25 a une variance ≈ 0,5²/12 ≈ 0,021, donc `√(0,85² + 0,021) ≈ 0,862`, soit environ +0,012.
  - Compromis : un **arrondi partiel** `r̃ = r̂ + α (round_θ(r̂) − r̂)` avec α ∈ [0, 1], réglé selon la pondération des deux métriques sur le classement.
  - Si la plateforme arrondit elle-même pour l'exactitude, on rend des valeurs continues et seule la calibration compte.
  - **À vérifier dans la doc de la plateforme** : qui arrondit, et comment.
- Le **clipping** dans [0,5 ; 5] est un gain gratuit (≈ −0,001 à −0,003).

---

## 8. Transport optimal et embedding laplacien (bref)

- **Transport optimal** : il n'existe pas de méthode OT standard compétitive en RMSE explicite. Les pistes existantes sont :
  - le **co-clustering par OT entropique** (Laclau, Iutzeler, Redko, ISI 2019, *Rank-one matrix factorization for co-clustering: an optimal transport perspective*) ;
  - le **matching via OT inverse** (Li et al., *Learning to Match via Inverse Optimal Transport*, JMLR 2019, arXiv:1802.03644), orienté contraintes d'offre ;
  - la **complétion distributionnelle par kNN dans l'espace de Wasserstein** (arXiv:2410.13112, 2024) ;
  - les **CF « géométriques » par perte de Sinkhorn** (arXiv:2410.03064, 2024), orientées implicite.
  - Intérêt pour nous : surtout pédagogique. Un co-clustering OT est comparable au `CoClustering` de Surprise (0,963 sur ML-100K), donc faible.
- **Embedding laplacien / régularisation de graphe** :
  - **Belkin, M., Niyogi, P. (2003).** *Laplacian Eigenmaps for Dimensionality Reduction and Data Representation.* Neural Computation 15(6).
  - **Cai, D., He, X., Han, J., Huang, T. (2011).** *Graph Regularized Nonnegative Matrix Factorization.* IEEE TPAMI 33(8).
  - **Kalofolias, V., Bresson, X., Bronstein, M., Vandergheynst, P. (2014).** *Matrix Completion on Graphs.* arXiv:1408.1717.
  - **Rao, N., Yu, H.-F., Ravikumar, P., Dhillon, I. (2015).** *Collaborative Filtering with Graph Information: Consistency and Scalable Methods (GRALS).* NIPS 2015.
  - Régularisation : `+ γ tr(Vᵀ L_G V)`, avec L_G le Laplacien d'un graphe de films (k-NN sur genres + année + TF-IDF du titre, ou sur co-notation). Cela lisse les facteurs des films voisins.
  - C'est un équivalent « graphe » du prior genre (§6.2) et du KMF (§6.5). Implémentation ALS : chaque mise à jour de V devient un système couplé, résolu par gradient conjugué.
  - Gain probable : faible et comparable à la FM + genres. C'est surtout utile si l'on veut une méthode « du cours ».

---

## 9. Benchmarks de référence

### 9.1 Surprise (https://surpriselib.com/, CV 5-fold, paramètres par défaut, CPU i5 11e gén.)

| Algorithme | ML-100K RMSE | MAE | Temps | ML-1M RMSE | MAE | Temps |
|---|---|---|---|---|---|---|
| SVD | 0,934 | 0,737 | 0:06 | 0,873 | 0,686 | 1:07 |
| SVD++ | **0,919** | **0,721** | 1:22-1:39 | **0,862** | **0,672** | 35-41 min |
| NMF | 0,963 | 0,758 | 0:06 | 0,916 | 0,723 | 1:39 |
| Slope One | 0,946 | 0,743 | 0:09 | 0,907 | 0,715 | 2:31 |
| k-NN | 0,980 | 0,774 | 0:08 | 0,923 | 0,727 | 5:27 |
| Centered k-NN | 0,951 | 0,749 | 0:09 | 0,929 | 0,738 | 5:43 |
| **k-NN Baseline** | 0,931 | 0,733 | 0:13 | 0,895 | 0,706 | 5:55 |
| Co-Clustering | 0,963 | 0,753 | 0:06 | 0,915 | 0,717 | 0:31 |
| Baseline (BaselineOnly) | 0,944 | 0,748 | 0:02 | 0,909 | 0,719 | 0:19 |
| Random | 1,518 | 1,219 | 0:01 | 1,504 | 1,206 | 0:19 |

Lecture :
- avec les paramètres par défaut, l'écart entre baseline et meilleur modèle n'est que de 0,025 (100K) à 0,047 (1M) ;
- SVD++ gagne environ 0,01-0,015 sur SVD ;
- KNNBaseline est proche de SVD sur 100K.
- Sur ml-latest-small, plus creux qu'ML-100K, l'écart baseline → meilleur modèle est typiquement **plus petit**. Nos 0,875 → 0,87 ne sont donc pas anormaux. Atteindre 0,80-0,815 sur la plateforme demande probablement l'effet train+test (§0.1), du MCMC et du blending.

### 9.2 Rendle et al. 2019
Voir le §1.5. Message clé : **Bayesian MF et Bayesian SVD++ via FM MCMC, bien réglés, battent les méthodes profondes** sur ML-10M (0,7563 contre 0,7634 pour le meilleur modèle publié).

---

## 10. Plan recommandé (classé)

Objectif : passer de ≈ 0,87 (local) à ≈ 0,83-0,85 en local, ce qui laisse espérer ≈ 0,80-0,83 sur la plateforme grâce au fit sur train+test.
**Protocole de validation à adopter d'abord** : K-fold (K = 5) sur train+test, qui reproduit le régime « plus de données » de la plateforme et fournit les prédictions out-of-fold pour le blend et les seuils d'arrondi.

### 1. Bayesian FM par MCMC (`myfm`) = BPMF + SVD++ + genres + année
- **Pourquoi** : c'est la méthode classique la plus forte sur petites données (Rendle 2019). La moyenne MCMC évite le sur-apprentissage qui plafonne notre MF SGD/ALS à 0,88, et les hyperparamètres sont échantillonnés, donc presque pas de réglage.
- **Features** : user, item, implicite utilisateur (films notés dans train+test, normalisés en `1/√n`), implicite item « flipped », genres en multi-hot normalisés, année en tranches. Groupes séparés.
- **Hyperparamètres** : rank 8-32, `n_iter` 300-1000, `n_kept_samples` ≈ `n_iter` − 50, init_stdev 0,1.
- **Gain attendu** : −0,02 à −0,035 par rapport à notre MF (estimation ; BPMF seul ≈ −0,015 à −0,02, puis implicite et side info ≈ −0,005 à −0,015).
- **Complexité** : faible (≈ 60-100 lignes de construction de matrices creuses ; `pip install myfm`).
- **Temps** : 10-60 s.
- **Risque** : disponibilité de wheel sur la plateforme. Le repli est un BPMF Gibbs maison en numpy (≈ 80 lignes, un peu moins bon sans implicite).

### 2. KNNBaseline item-based (Pearson-baseline rétréci), puis blend linéaire
- **Pourquoi** : c'est le modèle classique le plus décorrélé de la MF. Notre item-kNN à 0,890 manque baseline et shrinkage.
- **Hyperparamètres** : k = 30-50, `shrinkage=100`, `min_support=1-5`, baselines ALS (`reg_i=10, reg_u=15`). Réglage CV de k et du shrinkage.
- **Gain attendu** : seul, 0,86-0,865 en local ; en **blend** avec le n°1, −0,005 à −0,01 sur le n°1.
- **Complexité** : faible avec `scikit-surprise` (attention, wheels parfois absents pour Python récent ; sinon ≈ 60 lignes numpy, la matrice de similarité 4980² ≈ 200 Mo en float64 ou 100 Mo en float32).
- **Temps** : 5-30 s.

### 3. Blend ridge par K-fold stacking (+ méta-features simples)
- **Membres** : (a) FM MCMC du n°1, (b) KNNBaseline, (c) Soft-Impute ou ALS à **régularisation pondérée par fréquence** (trace-norm pondérée, §3.4), (d) baseline avec biais genre/année (§6.2), éventuellement (e) SVD++ SGD.
- **Gain attendu** : −0,005 à −0,015 sur le meilleur modèle seul (ordre de grandeur Netflix pour 3-5 modèles hétérogènes).
- **Complexité** : moyenne (orchestration K-fold, gestion du budget temps).
- **Temps** : K × Σ(temps des modèles). À 5 folds, tout doit tenir sous ≈ 1000 s. Il faut donc réduire les itérations MCMC dans les folds, ou fitter les poids sur un seul holdout.

### 4. Biais ancrés genre/année (prior hiérarchique) et ALS à λ pondéré
- **Pourquoi** : c'est quasi gratuit et cela améliore les films à 1-3 notes, très nombreux. C'est aussi un bon membre de blend, et ça sert de repli si myfm est indisponible.
- **Gain attendu** : −0,003 à −0,01 sur la baseline et le Soft-Impute actuels. Une partie redondante est absorbée par le n°1.
- **Complexité** : très faible. **Temps** : < 5 s.

### 5. Optimisation de l'exactitude : seuils appris / modèle ordinal
- **Pourquoi** : le mode conditionnel diffère de la moyenne et les entiers sont sur-représentés. On apprend des seuils de décision sur les prédictions out-of-fold, ou on utilise l'ordered probit de `myfm`.
- **Gain attendu** : +1 à +3 points d'exactitude par rapport à l'arrondi naïf au 0,5 (estimation). RMSE inchangé si la plateforme arrondit elle-même ; sinon, compromis via α (§7.2).
- **Complexité** : faible. **Temps** : négligeable.

**À ne pas prioriser** : SVT, max-norm, SLIM/EASE (implicite), LightFM (pertes de classement), timeSVD++ (pas de timestamps), OT, embeddings sentence-transformers (dépendance réseau ; gain < 0,005 ; seule option sûre : embeddings pré-calculés et commités, si le règlement l'autorise).

**Trajectoire réaliste** (local, K-fold sur train+test) : baseline ≈ 0,86 ; FM MCMC + implicite + genres ≈ 0,835-0,845 ; blend ≈ 0,83-0,84. Sur la plateforme, on peut espérer ≈ 0,81-0,83 (estimation, dépend fortement de la distribution du jeu eval caché).

---

## Sources principales (URL)
- Surprise benchmarks : https://surpriselib.com/
- Rendle, Zhang, Koren 2019 : https://arxiv.org/abs/1905.01395
- myFM : https://github.com/tohtsky/myFM
- Koren 2008 (KDD) : https://doi.org/10.1145/1401890.1401944
- Koren, Bell, Volinsky 2009 : https://doi.org/10.1109/MC.2009.263
- Koren 2009 timeSVD++ : https://doi.org/10.1145/1557019.1557072
- Salakhutdinov & Mnih 2007 / 2008 : NIPS 2007 ; https://doi.org/10.1145/1390156.1390267
- Mazumder, Hastie, Tibshirani 2010 : https://jmlr.org/papers/v11/mazumder10a.html
- Hastie et al. 2015 : https://jmlr.org/papers/v16/hastie15a.html
- Cai, Candès, Shen 2010 : https://doi.org/10.1137/080738970
- Salakhutdinov & Srebro 2010 (weighted trace norm) : NIPS 23
- Rendle 2010 / 2012 : https://doi.org/10.1109/ICDM.2010.127 ; https://doi.org/10.1145/2168752.2168771
- Singh & Gordon 2008 : https://doi.org/10.1145/1401890.1401969
- Kula 2015 (LightFM) : https://arxiv.org/abs/1507.08439
- Jain & Dhillon 2013 (IMC) : https://arxiv.org/abs/1306.0626
- Ning & Karypis 2011 (SLIM) : https://doi.org/10.1109/ICDM.2011.134
- Jahrer, Töscher, Legenstein 2010 (blending) : https://doi.org/10.1145/1835804.1835893
- OT : https://arxiv.org/abs/1802.03644 ; https://arxiv.org/abs/2410.13112 ; https://arxiv.org/abs/2410.03064
