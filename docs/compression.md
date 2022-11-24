# compression quality

$$
reconstruction \quad error = mean((X-\hat{X})^2)
$$

For more see the logs in the git history:

```
chr1, width 512
wandb:      Max(training reconstruction error) 0.06596
wandb:    Max(validation reconstruction error) 0.06674
wandb:     Mean(training reconstruction error) 0.04048
wandb: Mean(valilidation reconstruction error) 0.04102
wandb:      Min(training reconstruction error) 0.02018
wandb:    Min(validation reconstruction error) 0.02043
wandb:                                N models 96

chr1, width 16
wandb:      Max(training reconstruction error) 0.20218
wandb:    Max(validation reconstruction error) 0.20127
wandb:     Mean(training reconstruction error) 0.03809
wandb: Mean(valilidation reconstruction error) 0.03813
wandb:      Min(training reconstruction error) 0.00079
wandb:    Min(validation reconstruction error) 0.0007
wandb:                                N models 3084

chr2, width 512
wandb:      Max(training reconstruction error) 0.06197
wandb:    Max(validation reconstruction error) 0.06257
wandb:     Mean(training reconstruction error) 0.03966
wandb: Mean(valilidation reconstruction error) 0.04018
wandb:      Min(training reconstruction error) 0.01764
wandb:    Min(validation reconstruction error) 0.01788
wandb:                                N models 98

chr2, width 16
wandb:      Max(training reconstruction error) 0.23926
wandb:    Max(validation reconstruction error) 0.23852
wandb:     Mean(training reconstruction error) 0.03758
wandb: Mean(valilidation reconstruction error) 0.03762
wandb:      Min(training reconstruction error) 0.00103
wandb:    Min(validation reconstruction error) 0.00106
wandb:                                N models 3156

chr3, width 512
wandb:      Max(training reconstruction error) 0.0802
wandb:    Max(validation reconstruction error) 0.08107
wandb:     Mean(training reconstruction error) 0.04134
wandb: Mean(valilidation reconstruction error) 0.04188
wandb:      Min(training reconstruction error) 0.01142
wandb:    Min(validation reconstruction error) 0.01149
wandb:                                N models 82

chr3, width 16
wandb:      Max(training reconstruction error) 0.20604
wandb:    Max(validation reconstruction error) 0.20515
wandb:     Mean(training reconstruction error) 0.03766
wandb: Mean(valilidation reconstruction error) 0.03768
wandb:      Min(training reconstruction error) 0.00098
wandb:    Min(validation reconstruction error) 0.00095
wandb:                                N models 2646

chr4, width 512
wandb:      Max(training reconstruction error) 0.06998
wandb:    Max(validation reconstruction error) 0.07075
wandb:     Mean(training reconstruction error) 0.03995
wandb: Mean(valilidation reconstruction error) 0.04047
wandb:      Min(training reconstruction error) 0.01751
wandb:    Min(validation reconstruction error) 0.01774
wandb:                                N models 78

chr4, width 16
wandb:      Max(training reconstruction error) 0.17086
wandb:    Max(validation reconstruction error) 0.16949
wandb:     Mean(training reconstruction error) 0.03775
wandb: Mean(valilidation reconstruction error) 0.03776
wandb:      Min(training reconstruction error) 0.0012
wandb:    Min(validation reconstruction error) 0.00118
wandb:                                N models 2511

chr5


chr6

chr7

chr8

chr9

chr10

chr11

chr12

chr13

chr14

chr15

chr16

chr17

chr18

chr19

chr20

chr21

chr22

```