(::TruePositive)(ft::FairTensor) = sum(ft.mat[:, 1, 1])
(::FalsePositive)(ft::FairTensor) = sum(ft.mat[:, 1, 2])
(::FalseNegative)(ft::FairTensor) = sum(ft.mat[:, 2, 1])
(::TrueNegative)(ft::FairTensor) = sum(ft.mat[:, 2, 2])

# The functions true_positive, false_negative, etc are instances of TruePositive, FalseNegative, etc.
# So on using true_positive(fair_tensor) will use the above defined functions
(::TPR)(ft::FairTensor) = 1/(1+false_negative(ft)/true_positive(ft))
(::TNR)(ft::FairTensor) = 1/(1+false_positive(ft)/true_negative(ft))
(::FPR)(ft::FairTensor) = 1-true_negative_rate(ft)
(::FNR)(ft::FairTensor) = 1-true_positive_rate(ft)


(::FDR)(ft::FairTensor) = 1/(1+false_positive(ft)/true_positive(ft))
(::Precision)(ft::FairTensor) = 1 - false_discovery_rate(ft)
(::NPV)(ft::FairTensor) =  1/(1+false_negative(ft)/true_negative(ft))
