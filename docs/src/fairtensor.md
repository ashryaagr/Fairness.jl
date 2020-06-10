# Fairness Tensor

## Introduction

MLJFair uses the concept of Fairness Tensor to compute metrics and speed up the computation. In MLJFair, FairTensor is a struct with a 3D matrix and an array of strings for the class names in protected attribute. For a FairTensor `ft`, the 3D matrix can be accessed using `ft.mat` and the array of strings can be accessed using `ft.labels`.

`ft.mat` is a 3-dimensional Array. For a dataset with C number of classes in the sensitive attribute, a fairness tensor with matrix of size size C x 2 x 2 is constructed.

It is a stack of C 2-dimensional arrays of size 2 x 2 arrays. Each 2 x 2 array represents [[TP, FP], [FN, TN]]. Here TP corresponds to True Positives, FP to False Positives, FN to False Negatives and TN to True Negatives for each class in the protected attribute.

## Using Fairness Tensor


```@docs
MLJFair.FairTensor
fair_tensor
```

### Example

```@repl
using MLJFair
ŷ = categorical([1, 0, 1, 1, 0]);
y = categorical([0, 0, 1, 1, 1]);
grp = categorical(["Asian", "African", "Asian", "American", "African"]);
ft = fair_tensor(ŷ, y, grp);
ft.mat
ft.labels
```
