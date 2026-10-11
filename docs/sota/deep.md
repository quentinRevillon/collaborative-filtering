# État de l'art : deep learning pour la complétion de matrice à notes explicites (MovieLens, RMSE)

Contexte (notre cas) : 610 utilisateurs x 4980 films, environ 63k notes pour l'entraînement final (train+test), densité d'environ 2,1 %, notes de 0,5 à 5 (10 niveaux). En moyenne, environ 103 notes par utilisateur et environ 12,7 par film. Contraintes : CPU 4 threads, 12 Go, 1200 s au total.
À titre de comparaison, ML-100K a une densité de 6,3 % (943 x 1682) et ML-1M de 4,5 %. Notre matrice est 2 à 3 fois plus creuse, avec très peu de notes par film.

Vérification des sources : les chiffres marqués (P) ont été lus dans le PDF original. Ceux marqués (S) viennent d'une source secondaire (agrégateur ou tableau d'un autre papier).

## 1. Tableau de référence (RMSE, plus bas = mieux)

| Méthode | ML-100K (u1) | ML-1M (90/10) | ML-10M (90/10) | Source |
|---|---|---|---|---|
| BiasedMF (tel que rapporté dans les papiers deep) | – | 0.845 | 0.803 | AutoRec (P) |
| SGD-MF bien réglée (Rendle 2019) | – | – | **0.7720** | Rendle 2019 (P) |
| Bayesian MF (Gibbs, dim 512) | – | – | **0.7633** | Rendle 2019 (P) |
| Bayesian SVD++ / timeSVD++ / timeSVD++ flipped | – | – | 0.7563 / 0.7523 / **0.7485** | Rendle 2019 (P) |
| I-AutoRec (1 couche, 500 unités) | 0.895* | 0.831 | 0.782 | AutoRec (P), *Muller 2018 |
| I-AutoRec profond (500-250-500) | – | 0.827 | – | AutoRec (P) |
| CF-NADE (2 couches) | – | 0.829 | 0.771 | GC-MC / IGMC tables (P) |
| LLORMA | 0.898 | 0.833 | 0.7815 | (P) |
| GC-MC (sans side info) | 0.910 | 0.832 | 0.777 | GC-MC (S, ar5iv) / IGMC (P) |
| GC-MC + features | 0.905 | – | – | (P) |
| Sparse FC (1) / (2) | **0.890** / 0.894 | 0.830 / **0.824** | 0.784 / **0.769** | Muller 2018 (S, texte ICML) |
| IGMC | 0.905 | 0.857 | – | IGMC (P) |
| GraphRec / +Extra | 0.904 / 0.897 | 0.843 / 0.842 | – | GLocal-K (P) |
| MG-GAT (+side info) | 0.890 | – | – | GLocal-K (P) |
| GLocal-K | **0.890** (0.8889) | **0.822** (0.8227) | – | GLocal-K (P) |
| GHRS (+side info) | 0.887 | – | – | leaderboard HyperAI (S) |

Leaderboards Papers-with-Code (aujourd'hui recopiés par hyper.ai et sotaverified, car PwC a fermé) :
- ML-100K, RMSE sur le split u1 : GHRS 0.887 > GLocal-K 0.8889 > MG-GAT 0.890 > GraphRec 0.904 > IGMC 0.905 > GC-MC 0.905/0.910.
- ML-1M : GLocal-K 0.822 > Sparse FC 0.824 > CF-NADE 0.829 > I-AutoRec 0.831 > GC-MC 0.832.
- ML-10M : les mêmes méthodes (Sparse FC 0.769, CF-NADE 0.771) sont battues par les baselines bayésiennes de Rendle 2019 (0.7485).

## 2. Fiches par méthode

### AutoRec / I-AutoRec
- **Titre** : *AutoRec: Autoencoders Meet Collaborative Filtering*. S. Sedhain, A. K. Menon, S. Sanner, L. Xie. WWW 2015 Companion, pp. 111-112. https://ssanner.github.io/papers/anu/www15.pdf
- **Idée** : un autoencodeur à une couche cachée reconstruit le vecteur de notes d'un item (I-AutoRec). La loss n'est calculée que sur les entrées observées, avec une pénalité L2.
- **Réglages (P)** : k = 500 unités cachées, g = sigmoïde, f = identité (ReLU est moins bon). λ choisi dans {0.001, ..., 1000}. Optimiseur Rprop (aussi bon que L-BFGS et plus rapide). La note par défaut 3 est utilisée pour les users/items absents du train. Split 90/10, dont 10 % du train pris comme validation. Le RMSE baisse de façon monotone avec k (0.87 à k=10, 0.831 à k=500). I-AutoRec est bien meilleur que U-AutoRec sur ML-1M (0.831 contre 0.874).
- **RMSE (P)** : ML-1M 0.831, ML-10M 0.782, Netflix 0.823. Version profonde (500-250-500) : 0.827 sur ML-1M.
- **Coût** : faible. Avec un batch complet, une passe avant/arrière sur 4980 x 610 avec 500 unités coûte environ 10 GFLOP, soit à peu près 0,2-0,5 s par époque sur CPU 4 threads. 300 époques prennent moins de 2 min.
- **Verdict** : **faisable, environ 60 lignes de PyTorch.** Le gain seul est incertain : sur ML-10M, une MF bien réglée fait aussi bien. Mais ses erreurs sont décorrélées de celles de la MF, ce qui est utile en ensemble.
  Particularité de notre matrice : il y a 103 notes par user contre 12,7 par item. L'argument du papier en faveur de I-AutoRec (plus de notes par item que par user) s'inverse donc ici. Il faut **tester I- et U-AutoRec**.

### NCF / NeuMF et Rendle 2020
- **Titre** : *Neural Collaborative Filtering*. X. He, L. Liao, H. Zhang, L. Nie, X. Hu, T.-S. Chua. WWW 2017. https://arxiv.org/abs/1708.05031
- **Idée** : remplacer le produit scalaire par un MLP sur les embeddings concaténés (GMF + MLP = NeuMF). Le modèle est conçu pour le feedback implicite (log-loss, HR/NDCG).
- **Titre** : *Neural Collaborative Filtering vs. Matrix Factorization Revisited*. S. Rendle, W. Krichene, L. Zhang, J. Anderson. RecSys 2020. https://arxiv.org/abs/2005.09683
- **Résultat** : avec des hyperparamètres bien choisis, le produit scalaire bat nettement le MLP sur les mêmes données (ML-1M, Pinterest). Un MLP a du mal à apprendre un produit scalaire.
- **RMSE** : NCF ne rapporte aucun RMSE sur les notes explicites.
- **Verdict** : **à éviter.** C'est facile à coder, mais il n'y a aucun gain attendu par rapport à une MF ou un SVD++ bien régularisé, et le risque de sur-apprentissage est élevé à 2 % de densité.

### Deep Matrix Factorization (DMF)
- **Titre** : *Deep Matrix Factorization Models for Recommender Systems*. H.-J. Xue, X. Dai, J. Zhang, S. Huang, J. Chen. IJCAI 2017. https://www.ijcai.org/proceedings/2017/447
- **Idée** : deux tours MLP, l'une prend la ligne de la matrice (user), l'autre la colonne (item). La similarité cosinus des sorties est entraînée avec une cross-entropie « normalisée » par la note max. Les notes explicites et le feedback implicite sont combinés.
- **Évaluation** : top-K (HR@10, NDCG@10). Le papier ne donne pas de RMSE.
- **Verdict** : **peu pertinent.** La tour user ressemble à un U-AutoRec sans reconstruction. Un DMF régressif est faisable mais n'est pas validé en RMSE.

### CF-NADE
- **Titre** : *A Neural Autoregressive Approach to Collaborative Filtering*. Y. Zheng, B. Tang, W. Ding, H. Zhou. ICML 2016. https://arxiv.org/abs/1605.09477
- **Idée** : NADE (variante autorégressive des RBM) appliqué aux notes. On modélise P(note) comme une distribution sur 5 classes, avec partage des poids entre notes ordinales et ordres aléatoires des items.
- **Réglages** : 500 unités cachées.
- **RMSE** : ML-1M 0.830 avec 1 couche, 0.829 avec 2 couches. ML-10M 0.771. Netflix 0.803.
- **Coût** : modéré, mais l'implémentation est délicate (ordonnancement aléatoire, softmax par niveau de note ; nous avons 10 niveaux).
- **Verdict** : **faisable mais plus de 100 lignes et fragile.** Le gain par rapport à AutoRec est de l'ordre de 0.002 RMSE sur ML-1M : mauvais rapport effort/bénéfice.

### Sparse FC (Kernelized Synaptic Weight Matrices)
- **Titre** : *Kernelized Synaptic Weight Matrices*. L. Muller, J. Martel, G. Indiveri. ICML 2018, PMLR 80:3654-3663. https://proceedings.mlr.press/v80/muller18a.html
- **Idée** : on garde l'architecture I-AutoRec, mais chaque poids W_ij est multiplié par un noyau à support fini K(u_i, v_j) = max(0, 1 - ||u_i - v_j||²). Les u et v sont des vecteurs appris de dimension 5. On obtient ainsi une sparsification apprise des connexions, qui sert de régularisation.
- **Réglages (S)** : couches cachées de 500, d = 5. Pénalités L2 λ₂ sur W et λ₀ sur le noyau. Optimiseur L-BFGS-B ou Rprop. Validation : 20 % sur ML-100K, 10 % sur ML-1M.
- **RMSE (S)** : ML-100K 0.890 (1 couche), ML-1M 0.824 (2 couches), ML-10M 0.769.
- **Verdict** : **faisable, c'est I-AutoRec plus environ 15 lignes.** C'est le cœur de GLocal-K.

### GLocal-K
- **Titre** : *GLocal-K: Global and Local Kernels for Recommender Systems*. S. C. Han, T. Lim, S. Long, B. Burgstaller, J. Poon. CIKM 2021 (short). https://arxiv.org/abs/2108.12184. Code TF1 : https://github.com/usydnlp/Glocal_K. Port PyTorch : https://github.com/fleanend/TorchGlocalK
- **Idée** : deux étapes.
  1. Pré-entraîner un Sparse-FC AE (« noyau local » RBF à support fini).
  2. Construire un « noyau global » 3x3 : moyenne des sorties item du modèle pré-entraîné, puis combinaison de m noyaux 3x3 appris. On le convolue sur la matrice de notes (conv2d, ReLU), puis on fine-tune l'AE sur cette matrice filtrée.
- **RMSE (P)** : ML-100K 0.890 (0.8889), ML-1M 0.822 (0.8227), Douban 0.721.
- **Hyperparamètres (code officiel)** : n_hid = 500, n_dim = 5, n_layers = 2, gk_size = 3, sigmoïde en couche cachée, identité en sortie, L-BFGS-B.

  | Paramètre | ML-100K | ML-1M | Douban |
  |---|---|---|---|
  | lambda_2 | 20 | 70 | 10 |
  | lambda_s | 0.006 | 0.018 | 0.022 |
  | iter_p | 5 | 50 | 5 |
  | iter_f | 5 | 10 | 5 |
  | epoch_p | 30 | 20 | 20 |
  | epoch_f | 60 | 30 | 60 |
  | dot_scale | 1 | 0.5 | 2 |

  La matrice d'entrée est items x users. Les prédictions sont clippées à [1, 5] (pour nous, [0,5 ; 5]).
- **Points d'attention** :
  1. Le notebook officiel garde la **meilleure époque mesurée sur le test**. Le 0.889 est donc optimiste : il faut sélectionner l'époque sur une validation.
  2. La Fig. 2 du papier, sur ML-100K, montre une forte dégradation quand on réduit les données d'entraînement : environ 0.95 à 40 % et environ 1.03 à 20 % (avec un comportement identique pour Sparse FC). Notre densité de 2,1 % correspond à environ 33 % de ML-100K, d'où un RMSE attendu proche de 0.95-0.97 sur cette échelle. L'AE transductif souffre de la sparsité.
- **Coût** : environ 4980 x 610 x 500 par couche, L-BFGS full-batch. Avec environ 90 « époques » x 5 itérations L-BFGS, on reste en général sous 2-4 min sur CPU.
- **Verdict** : **faisable en environ 120 lignes de PyTorch.** C'est la méthode deep sans side info la plus forte sur ML-100K et ML-1M. Le gain réel par rapport à une bonne MF ou SVD++ reste à mesurer : prévoir une validation stricte et un blend.

### GC-MC
- **Titre** : *Graph Convolutional Matrix Completion*. R. van den Berg, T. N. Kipf, M. Welling. arXiv 1706.02263 (2017), KDD Deep Learning Day 2018. https://arxiv.org/abs/1706.02263
- **Idée** : autoencodeur de graphe sur le graphe biparti user-item. On fait un message passing avec un poids par niveau de note, puis un décodeur bilinéaire qui donne un softmax sur les notes.
- **Réglages (S)** : couche GC de 500 unités (ReLU), couche dense de 75, dropout 0.7, Adam avec lr 0.01. Batch complet : 1000 époques sur ML-100K, 3500 sur ML-1M. 2 matrices de base dans le décodeur. Accumulation « stack » sur ML-100K, « sum » sur ML-1M.
- **RMSE** : ML-100K 0.910 (0.905 avec features), ML-1M 0.832, ML-10M 0.777. Sur Flixster (notes 0,5-5, densité 0,29 %) : 0.917.
- **Coût CPU** : faible. Avec des matrices creuses de 63k arêtes, 1000 époques prennent quelques minutes.
- **Verdict** : faisable en environ 120-150 lignes. **Moins bon qu'AutoRec ou GLocal-K** sur MovieLens. Avec nos 10 niveaux de notes, il y a beaucoup de paramètres par niveau. Intérêt faible.

### IGMC
- **Titre** : *Inductive Matrix Completion Based on Graph Neural Networks*. M. Zhang, Y. Chen. ICLR 2020. https://arxiv.org/abs/1904.12058. Code : https://github.com/muhanzhang/IGMC
- **Idée** : on extrait le sous-graphe à 1 saut autour de chaque paire (u, i), on applique 4 couches R-GCN (32 dimensions) avec étiquetage des nœuds, puis un MLP qui régresse la note. Le modèle est inductif et n'utilise pas d'ID.
- **Réglages (P)** : 4 couches R-GCN de 32, 4 bases, MLP de 128 avec dropout 0.5, edge dropout 0.2, ARR λ = 0.001, Adam lr 1e-3, batch 50. 80 époques sur ML-100K, avec un ensemble des époques 50/60/70/80.
- **RMSE (P)** : ML-100K 0.905, ML-1M 0.857, Flixster 0.872, Douban 0.721.
  Point fort : sa Fig. 2 montre une bien **meilleure robustesse à la sparsité** que GC-MC sur ML-1M. Avec 1 % des notes, IGMC reste vers 0.95 alors que GC-MC dépasse 1.05.
- **Coût** : élevé. Il faut un sous-graphe par note, soit 63k sous-graphes par époque. Les films populaires ont des centaines de voisins, donc il faut sous-échantillonner. Il faut aussi torch_geometric. Sur 4 threads CPU, plusieurs dizaines d'époques ne tiennent probablement pas dans 1200 s.
- **Verdict** : **difficile dans le budget**, et pas en 100 lignes. Ce serait intéressant pour la sparsité, mais trop risqué.

### LightGCN, Mult-VAE (VAE-CF), BERT4Rec
- **LightGCN** : *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*. X. He et al. SIGIR 2020. https://arxiv.org/abs/2002.02126. Propagation linéaire d'embeddings, perte BPR, implicite uniquement (Recall/NDCG).
  Verdict : on pourrait adapter son lissage de graphe à la régression (MSE), mais aucun résultat RMSE n'est publié. Faible priorité.
- **Mult-VAE** : *Variational Autoencoders for Collaborative Filtering*. D. Liang, R. G. Krishnan, M. D. Hoffman, T. Jebara. WWW 2018. https://arxiv.org/abs/1802.05814. VAE multinomial sur l'historique de clics, implicite uniquement.
  Verdict : non pertinent pour le RMSE.
- **BERT4Rec** : *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*. F. Sun et al. CIKM 2019. https://arxiv.org/abs/1904.06690. Prédiction du prochain item dans une séquence.
  Verdict : non pertinent (pas de régression de note, données trop petites).

### Rendle, Zhang, Koren 2019 : la difficulté d'évaluer les baselines
- **Titre** : *On the Difficulty of Evaluating Baselines: A Study on Recommender Systems*. S. Rendle, L. Zhang, Y. Koren. arXiv 1905.01395 (2019). https://arxiv.org/abs/1905.01395
- **Résultats (P)**, sur ML-10M avec split 90/10 :
  - SGD-MF : 0.7720 en dimension 512, 0.7756 en dimension 64.
  - Bayesian MF (Gibbs, dim 512) : 0.7633.
  - Bayesian timeSVD : 0.7587, Bayesian SVD++ : 0.7563, Bayesian timeSVD++ : 0.7523.
  - **Bayesian timeSVD++ flipped : 0.7485.** C'est mieux que toutes les méthodes publiées, y compris I-AutoRec 0.782, CF-NADE 0.771, Sparse FC 0.769 et MRMA 0.7634.
- **Leçon** : sur la complétion explicite, les gains « deep » rapportés viennent en grande partie de baselines mal réglées. Le feedback implicite (SVD++, côté user et côté item, « flipped ») et l'inférence bayésienne comptent plus que l'architecture.

## 3. Synthèse pour notre cadre

- Notre densité est d'environ 2 %, avec en moyenne 12,7 notes par film. Les AE transductifs (AutoRec, Sparse FC, GLocal-K) se dégradent fortement à cette sparsité (Fig. 2 de GLocal-K). IGMC résiste mieux mais est trop coûteux sur CPU.
- Le meilleur pari « deep » reste un **AutoRec régularisé ou noyauté**, utilisé **en ensemble** avec des modèles MF, SVD++ ou bayésiens. Ces derniers devraient rester le socle d'après Rendle 2019.
- Écart attendu : sur ML-100K et ML-1M, deep contre BiasedMF publié donne -0.01 à -0.02 de RMSE. Contre une MF bien réglée, l'écart est proche de 0. Le gain par blend est en général de 0.005 à 0.015.

## 4. Recommandation classée

1. **I-AutoRec / U-AutoRec, puis variante Sparse FC (noyau local).** Rapport coût/gain le plus sûr. Hyperparamètres :
   - 500 unités cachées, sigmoïde puis identité, 1 couche (essayer 2 couches 500-500).
   - Loss MSE masquée et pénalité L2 λ/2·(‖W‖²+‖V‖²), avec λ autour de 1 à 100 à cause de la normalisation par lot (GLocal-K utilise λ₂ = 20 sur ML-100K).
   - L-BFGS full-batch ou Rprop. Avec Adam : lr 1e-3, 200 à 500 époques.
   - Note par défaut : moyenne globale (ou baseline) pour les users/items inconnus. Clip à [0,5 ; 5].
   - Pour le noyau local : u, v de dimension 5, λ_s environ 0.006-0.02.
   - Tester les deux orientations : notre matrice compte 103 notes par user contre 12,7 par item.
   - Temps estimé : moins de 2 min sur CPU.
2. **GLocal-K complet** : étape 1 ci-dessus, puis noyau global 3x3 et fine-tuning.
   - Hyperparamètres ML-100K : n_hid 500, n_dim 5, 2 couches, gk 3x3, λ₂ = 20, λ_s = 0.006, L-BFGS-B, 30 époques de pré-entraînement x 5 itérations, 60 époques de fine-tuning x 5 itérations, dot_scale 1.
   - Choisir l'époque **sur une validation, pas sur le test**.
   - Temps estimé : 2 à 5 min.
   - Gain par rapport au point 1 : environ -0.005 sur ML-100K/1M. Incertain à notre densité.
3. **(Optionnel) GC-MC léger** comme membre d'ensemble décorrélé.
   - 500 puis 75 unités, dropout 0.7, Adam avec lr 0.01, environ 1000 époques full-batch, 2 bases de décodeur, une matrice par niveau de note (10 niveaux, ou regroupés en 5).
   - Gain seul probablement nul par rapport à la MF.

À éviter : NCF/NeuMF (Rendle 2020), DMF (top-K), CF-NADE (complexe pour +0.002), IGMC (coût CPU), et LightGCN, Mult-VAE et BERT4Rec (implicites ou séquentiels).

Dans tous les cas, **blender avec une MF ou un SVD++ bien réglé** (voire une MF bayésienne), qui reste le meilleur prédicteur seul selon Rendle 2019.
