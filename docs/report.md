
# Report
Status on the project so far.

## Project
Central idea of the project.

- Is it possible to meaningfully compress single nucleotide polymorphism (SNP) data?
  - Is there a potential benefit for using a non-linear compression

## Single cSNP analysis
A single SNP analysis performed on compressed SNP (cSNPs).

![](images/single_snp_analysis/chr1-22_20k_identity_16_c_snps_train.quant.png)

![](images/single_snp_analysis/chr1-22_20k_identity_512_c_snps_train.quant.png)

![](images/single_snp_analysis/chr1-22_20k_relu_16_c_snps_train.quant.png)

![](images/single_snp_analysis/chr1-22_20k_relu_512_c_snps_train.quant.png)


## Single SNP analysis

![](images/single_snp_analysis/geno.quant.png)


### Comparison

| Activation | Stride/width | Compression factor | N significant (p < 5x10^-8) | Expected N given number of SNPs | N significant / expected |
| ---------- | ------------ | ------------------ | --------------------------- | ------------------------------- | ------------------------ |
| Identity   | 16           | 2x                 | 29                          | 0.013748                        | **2109.2903**            |
| Identity   | 512          | 2x                 | 7                           | 0.015503                        | 451.4963                 |
| ReLU       | 16           | 2x                 | 20                          | 0.013748                        | 1454.6829                |
| ReLU       | 512          | 2x                 | 1                           | 0.015503                        | 64.4994                  |
|            |              | Uncompressed       | 50                          | 0.03143                         | 1590.5989                |


<details>
<summary>Raw output</summary>
```bash
chr1-22_20k_identity_16_c_snps_train.quant - 0 / 4
        N significant (p < 5x10^-8): 29
        Expected N given number of SNPs: 0.013748699999999999
        N significant / expected:  2109.290332904202
chr1-22_20k_identity_512_c_snps_train.quant - 1 / 4
        N significant (p < 5x10^-8): 7
        Expected N given number of SNPs: 0.015503999999999999
        N significant / expected:  451.4963880288958
chr1-22_20k_relu_16_c_snps_train.quant - 2 / 4
        N significant (p < 5x10^-8): 20
        Expected N given number of SNPs: 0.013748699999999999
        N significant / expected:  1454.6829882097945
chr1-22_20k_relu_512_c_snps_train.quant - 3 / 4
        N significant (p < 5x10^-8): 1
        Expected N given number of SNPs: 0.015503999999999999
        N significant / expected:  64.49948400412798
geno.quant - 4 / 5
        N significant (p < 5x10^-8): 50
        Expected N given number of SNPs: 0.031434699999999996
        N significant / expected:  1590.598924118888
```
</details>


## REML

Analysis performed on Chromosome 1-22 using 20k participants derived from 4 different compression:

| Input size | Activation Function | Compression factor (intended) | Heritability |
| ---------- | ------------------- | ----------------------------- | ------------ |
| 16         | Identity            | 2x                            | 0.60         |
| 512        | Identity            | 2x                            | 0.54         |
| 16         | ReLU                | 2x                            | 0.59         |
| 512        | ReLU                | 2x                            | 0.55         |
|            |                     | Uncompressed                  | **0.82**     |



For more details you can inspect the following outputs from LDAK.
The model was performed under the GCTA model.


<details>
    <summary> LDAK Outputs </summary>

chr1-22_20k_identity_16_c_snps_train.reml1.reml
```bash
Num_Kinships 1
Num_Regions 0
Num_Top_Predictors 0
Num_Covariates 1
Num_Environments 0
Blupfile /home/kce/NLPPred/github/snip/data/ldak_results/chr1-22_20k_identity_16_c_snps_train.reml1.indi.blp
Regfile none
Coeffsfile /home/kce/NLPPred/github/snip/data/ldak_results/chr1-22_20k_identity_16_c_snps_train.reml1.coeff
Covar_Heritability 0.0000
Total_Samples 12862
With_Phenotypes 12862
Converged YES
Null_Likelihood -18297.3966
Alt_Likelihood -18054.2041
LRT_Stat 486.3851
LRT_P 4.3587e-108
Component Heritability SD Size Mega_Intensity SD
Her_K1 0.604960 0.027899 274974.00 2.200062 0.101461
Her_Top 0.000000 NA NA NA NA
Her_All 0.604960 0.027899 274974.00 2.200062 0.101461
```

chr1-22_20k_identity_512_c_snps_train.reml1.reml
```
Num_Kinships 1
Num_Regions 0
Num_Top_Predictors 0
Num_Covariates 1
Num_Environments 0
Blupfile /home/kce/NLPPred/github/snip/data/ldak_results/chr1-22_20k_identity_512_c_snps_train.reml1.indi.blp
Regfile none
Coeffsfile /home/kce/NLPPred/github/snip/data/ldak_results/chr1-22_20k_identity_512_c_snps_train.reml1.coeff
Covar_Heritability 0.0000
Total_Samples 12862
With_Phenotypes 12862
Converged YES
Null_Likelihood -18297.3966
Alt_Likelihood -18065.9040
LRT_Stat 462.9853
LRT_P 5.3854e-103
Component Heritability SD Size Mega_Intensity SD
Her_K1 0.544225 0.026037 310080.00 1.755113 0.083969
Her_Top 0.000000 NA NA NA NA
Her_All 0.544225 0.026037 310080.00 1.755113 0.083969
```


chr1-22_20k_relu_16_c_snps_train.reml1.reml
```
Num_Kinships 1
Num_Regions 0
Num_Top_Predictors 0
Num_Covariates 1
Num_Environments 0
Blupfile /home/kce/NLPPred/github/snip/data/ldak_results/chr1-22_20k_relu_16_c_snps_train.reml1.indi.blp
Regfile none
Coeffsfile /home/kce/NLPPred/github/snip/data/ldak_results/chr1-22_20k_relu_16_c_snps_train.reml1.coeff
Covar_Heritability 0.0000
Total_Samples 12862
With_Phenotypes 12862
Converged YES
Null_Likelihood -18297.3966
Alt_Likelihood -18068.0603
LRT_Stat 458.6727
LRT_P 4.6741e-102
Component Heritability SD Size Mega_Intensity SD
Her_K1 0.587304 0.027586 188901.00 3.109058 0.146037
Her_Top 0.000000 NA NA NA NA
Her_All 0.587304 0.027586 188901.00 3.109058 0.146037
```


chr1-22_20k_relu_512_c_snps_train.reml1.reml
```
Num_Kinships 1
Num_Regions 0
Num_Top_Predictors 0
Num_Covariates 1
Num_Environments 0
Blupfile /home/kce/NLPPred/github/snip/data/ldak_results/chr1-22_20k_relu_512_c_snps_train.reml1.indi.blp
Regfile none
Coeffsfile /home/kce/NLPPred/github/snip/data/ldak_results/chr1-22_20k_relu_512_c_snps_train.reml1.coeff
Covar_Heritability 0.0000
Total_Samples 12862
With_Phenotypes 12862
Converged YES
Null_Likelihood -18297.3966
Alt_Likelihood -18071.5190
LRT_Stat 451.7553
LRT_P 1.4965e-100
Component Heritability SD Size Mega_Intensity SD
Her_K1 0.553665 0.026147 100962.00 5.483895 0.258981
Her_Top 0.000000 NA NA NA NA
Her_All 0.553665 0.026147 100962.00 5.483895 0.258981
```

And based on the raw uncompressed SNPs:
```
Num_Kinships 1
Num_Regions 0
Num_Top_Predictors 0
Num_Covariates 1
Num_Environments 0
Blupfile /home/kce/NLPPred/github/snip/data/ldak_results/geno.reml1.indi.blp
Regfile none
Coeffsfile /home/kce/NLPPred/github/snip/data/ldak_results/geno.reml1.coeff
Covar_Heritability 0.0000
Total_Samples 12862
With_Phenotypes 12862
Converged YES
Null_Likelihood -18297.3966
Alt_Likelihood -18055.7829
LRT_Stat 483.2274
LRT_P 2.1205e-107
Component Heritability SD Size Mega_Intensity SD
Her_K1 0.824049 0.037624 628028.73 1.312120 0.059908
Her_Top 0.000000 NA NA NA NA
Her_All 0.824049 0.037624 628028.73 1.312120 0.05990
```


</details>


This assumes the model:

$$
y = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n + \epsilon = X\beta + \epsilon
$$

Where:
$$
\beta_j \sim N(0, K\sigma^2)
$$
    
$$
\epsilon \sim N(0, \sigma^2_\epsilon)
$$

Where $K$ is the kinship matrix.

We can thus describe $y$ as:

$$
y \sim N(\beta_0, K\sigma^2 + I\sigma^2_\epsilon)
$$

Where $K$ is the kinship matrix and $\epsilon$ is the residual error.

Heritability is then defined as:

$$
h^2 = \frac{\sigma^2}{\sigma^2 + \sigma^2_\epsilon}
$$


# compression quality

The compression is performed using a simple autoencoder with 3 hidden layers. All
networks use a compression factor of 2x and are run independently for each chromosome on
using a width and stride of 16 or 512 and using either a ReLU or Identity (PCA) activation function.

Reconstruction error is calucated as follows:
$$
reconstruction \quad error = mean((X-\hat{X})^2)
$$


<details>
    <summary> Raw outputs </summary>

```bash
chr1, width 512
Max(training reconstruction error) 0.06596
Max(validation reconstruction error) 0.06674
Mean(training reconstruction error) 0.04048
Mean(valilidation reconstruction error) 0.04102
Min(training reconstruction error) 0.02018
Min(validation reconstruction error) 0.02043
N models 96

chr1, width 16
Max(training reconstruction error) 0.20218
Max(validation reconstruction error) 0.20127
Mean(training reconstruction error) 0.03809
Mean(valilidation reconstruction error) 0.03813
Min(training reconstruction error) 0.00079
Min(validation reconstruction error) 0.0007
N models 3084

chr2, width 512
Max(training reconstruction error) 0.06197
Max(validation reconstruction error) 0.06257
Mean(training reconstruction error) 0.03966
Mean(valilidation reconstruction error) 0.04018
Min(training reconstruction error) 0.01764
Min(validation reconstruction error) 0.01788
N models 98

chr2, width 16
Max(training reconstruction error) 0.23926
Max(validation reconstruction error) 0.23852
Mean(training reconstruction error) 0.03758
Mean(valilidation reconstruction error) 0.03762
Min(training reconstruction error) 0.00103
Min(validation reconstruction error) 0.00106
N models 3156

chr3, width 512
Max(training reconstruction error) 0.0802
Max(validation reconstruction error) 0.08107
Mean(training reconstruction error) 0.04134
Mean(valilidation reconstruction error) 0.04188
Min(training reconstruction error) 0.01142
Min(validation reconstruction error) 0.01149
N models 82

chr3, width 16
Max(training reconstruction error) 0.20604
Max(validation reconstruction error) 0.20515
Mean(training reconstruction error) 0.03766
Mean(valilidation reconstruction error) 0.03768
Min(training reconstruction error) 0.00098
Min(validation reconstruction error) 0.00095
N models 2646

chr4, width 512
Max(training reconstruction error) 0.06998
Max(validation reconstruction error) 0.07075
Mean(training reconstruction error) 0.03995
Mean(valilidation reconstruction error) 0.04047
Min(training reconstruction error) 0.01751
Min(validation reconstruction error) 0.01774
N models 78

chr4, width 16
Max(training reconstruction error) 0.17086
Max(validation reconstruction error) 0.16949
Mean(training reconstruction error) 0.03775
Mean(valilidation reconstruction error) 0.03776
Min(training reconstruction error) 0.0012
Min(validation reconstruction error) 0.00118
N models 2511

And a similar tendency is seen from 5-22.
```

To see all check the logs in the git history.

</details>


# Question by Doug and answers

- What is the role of the kinship matrix in the model? (what is the role of $K$?)

The role of the kinship matrix $K$ is to capture the genetic relatedness between
individuals. This is important because we want to capture the genetic relatedness
between individuals in our model. If we don't capture this, we will get a biased
estimate of the heritability.

- How do you specify the kinship matrix in the model? (how do you set $K$?)

Using the flag `--grm` in `ldak` we can specify the kinship matrix. This is
calculated using the `ldak --calc-tagging` command.

- What did we use for the kinship matrix? (what is the role of $K$?)
  
We use the kinship matrix calculated from the data using the `ldak --calc-tagging`
command with `--ignore-weights YES` (I am a bit unsure of what this does - the `--power`
already provided the wieghting - is this an additional weight before the GCTA weighting) 
and `--power -1` (GCTA / naive). I am a bit unsure how this is done.

# TODO

- [x] Do a single snp analysis with the same number of individuals
  - Added to the report under the section "single snp analysis"
- [ ] Check relu compression (why is there trivial SNPs?)
- [ ] Redo analysis with more ind.
  - [ ]  time estimate (for compression?)
  - [ ]  more traits (e.g. Doug Speed will send a path for the bloodsamples)
- [ ] compare w. R^2 pruning
- [x] Plot Single SNPs
- [x] Figure out the role of the Q_J (examine)
- [ ] Calculate correlation
- [x] Check visualization, is there a even amount of c snps pr. chromosome?
  -  It was caused by a normalization of the snps. I should have updated the plots now.


## central question:
The fundamental question we are asking is, "are these compressions useful". Do they provide any benefits over having SNP data? So please think how we can answer this question. Ultimately, it is ok if the answer is "no", provided we are confident we have tried hard enough.  