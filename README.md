# DTANet-KAN for drug–target affinity prediction

This undergraduate project explores a modification to the drug and protein encoders in a model based on [DTANet+](https://doi.org/10.1007/s12539-026-00851-6). The experiments used Python and PyTorch.

## My contribution

I proposed the encoder changes and independently designed the 2 × 2 KIBA ablation. I used generative AI to assist with code implementation. In the KAN variant, Kolmogorov–Arnold network layers replace the position-wise multilayer perceptrons (MLPs) in the multi-kernel convolutional branches of both encoders. The convolutions, cross-attention and final affinity regressor are retained.

## KIBA encoder ablation

I compared the baseline and KAN variant on 50,000 drug–target pairs and on the full dataset. Each configuration used a random 85/15 training/validation split (split seed 42). The table shows the highest validation concordance index (CI) over 20 epochs in one recorded run per configuration, with mean squared error (MSE) from the same epoch.

| Dataset | Encoder | Best validation CI | MSE at that epoch |
| --- | --- | ---: | ---: |
| 50,000 pairs | Baseline | 0.8105 | 0.3169 |
| 50,000 pairs | KAN | 0.8239 | 0.2911 |
| Full | Baseline | 0.8379 | 0.2520 |
| Full | KAN | 0.8462 | 0.2327 |

The observed CI differences were +0.0134 and +0.0083, respectively. The random split evaluates held-out pairs. Generalization to unseen targets and variation across training seeds remain open questions.

See the [experiment notes](KIBA_ABLATION.md) and [validation summary](kiba_validation_summary.csv). Each archived training script and its epoch-by-epoch curve share a filename prefix, for example [the 50,000-pair KAN script](kiba_50k_kan_train.py) and [its curve](kiba_50k_kan_curve.csv). Running these scripts requires the project modules, separately obtained inputs and the paths described in the scripts.

## Follow-up work

In a separate experiment, I compared a KAN scoring head with an MLP of the same size. Training settings for both heads were selected within the training data before scoring one development fold (seed 42). Each drug contributed equally to the reported MSE.

| Method | Drug-weighted MSE (equal weight per drug) |
| --- | ---: |
| Fusion KAN after training-set selection | 0.7221 |
| MLP after training-set selection | 0.7289 |
| KronRLS reference | 0.7163 |

On this fold, the KAN head had lower error than the MLP, and KronRLS had the lowest error. These preliminary values concern a different model component and evaluation protocol from the KIBA encoder ablation above.

## Related material

- [Original DTANet+ implementation](https://github.com/202324131016T/DTANet_plus)
- [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756)
- [Archived three-way comparison figure](results/kiba_three_way_comparison.png)
