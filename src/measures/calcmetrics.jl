"
    _ftIdx(ft, grp)

Finds the index of grp (string) in ft.labels which corresponds to ft.mat.
For Index i for the grp  returned by this function ft[i, :, :] returns the
2D array [[TP, FP], [FN, TN]] for that group.
"
function _ftIdx(ft::FairTensor, grp)
    idx = findfirst(x->x==string.(grp), ft.labels)
    if idx==nothing throw(ArgumentError("$grp not found in the fairness tensor")) end
    return idx
end

# Helper function for calculating TruePositive, FalseNegative, etc.
# If grp is :, it calculates combined value for all groups, else the specified group
_calcmetric(ft::FairTensor, grp, inds...) = typeof(grp)==Colon ? sum(ft[:, inds...]) : sum(ft[_ftIdx(ft, grp), inds...])

(::TruePositive)(ft::FairTensor; grp=:) = _calcmetric(ft, grp, 1, 1)
(::FalsePositive)(ft::FairTensor; grp=:) = _calcmetric(ft, grp, 1, 2)
(::FalseNegative)(ft::FairTensor; grp=:) = _calcmetric(ft, grp, 2, 1)
(::TrueNegative)(ft::FairTensor; grp=:) = _calcmetric(ft, grp, 2, 2)


# The functions true_positive, false_negative, etc are instances of TruePositive, FalseNegative, etc.
# So on using true_positive(fair_tensor) will use the above defined functions
(::TPR)(ft::FairTensor; grp=:) = 1/(1+false_negative(ft; grp=grp)/(1e-15 + true_positive(ft; grp=grp)))
(::TNR)(ft::FairTensor; grp=:) = 1/(1+false_positive(ft; grp=grp)/(1e-15 + true_negative(ft; grp=grp)))
(::FPR)(ft::FairTensor; grp=:) = 1-true_negative_rate(ft; grp=grp)
(::FNR)(ft::FairTensor; grp=:) = 1-true_positive_rate(ft; grp=grp)


(::FDR)(ft::FairTensor; grp=:) = 1/(1+false_positive(ft; grp=grp)/(1e-15 + true_positive(ft; grp=grp)))
(::Precision)(ft::FairTensor; grp=:) = 1 - false_discovery_rate(ft; grp=grp)
(::NPV)(ft::FairTensor; grp=:) =  1/(1+false_negative(ft; grp=grp)/(1e-15 + true_negative(ft; grp=grp)))

(::MLJBase.Accuracy)(ft::FairTensor; grp=:) = (true_positive(ft; grp = grp) + true_negative(ft; grp = grp))/(
        true_positive(ft; grp = grp) + true_negative(ft; grp = grp) +
        false_positive(ft; grp = grp) + false_negative(ft; grp = grp))


struct PredictedPositiveRate <: Measure end
PP = PredictedPositiveRate
predicted_positive_rate = PP()
ppr = predicted_positive_rate
(::PP)(ft::FairTensor; grp=:) = _calcmetric(ft, grp, 1, :)/_calcmetric(ft, grp, :, :)

struct FalseOmissionRate <: Measure end
FOR = FalseOmissionRate
false_omission_rate = FOR()
foar = false_omission_rate
(::FOR)(ft::FairTensor; grp=:) = 1/(1+true_negative(ft; grp=grp)/(1e-15 + false_negative(ft; grp=grp)))

struct TruePositiveRateDifference <: Measure end
TPRD = TruePositiveRateDifference
true_positive_rate_difference = TPRD()
tprd = true_positive_rate_difference
(::TPRD)(ft::FairTensor; grp1=:,grp2=:)=abs(true_positive_rate(ft;grp=grp1)-true_positive_rate(ft;grp=grp2))

struct FalsePositiveRateDifference <: Measure end
FPRD = FalsePositiveRateDifference
false_positive_rate_difference = FPRD()
fprd = false_positive_rate_difference
(::FPRD)(ft::FairTensor; grp1=:,grp2=:)=abs(false_positive_rate(ft;grp=grp1)-false_positive_rate(ft;grp=grp2))

struct FalseNegativeRateDifference <: Measure end
FNRD = FalseNegativeRateDifference
false_negative_rate_difference = FNRD()
fnrd = false_negative_rate_difference
(::FNRD)(ft::FairTensor; grp1=:, grp2=:)=abs(false_negative_rate(ft;grp=grp1)-false_negative_rate(ft;grp=grp2))

struct FalseOmissionRateDifference <: Measure end
FORD = FalseOmissionRateDifference
false_omission_rate_difference = FORD()
ford = false_omission_rate_difference
(::FORD)(ft::FairTensor; grp1=:, grp2=:)=abs(false_omission_rate(ft;grp=grp1)-false_omission_rate(ft;grp=grp2))

struct FalseDiscoveryRateDifference <: Measure end
FDRD = FalseDiscoveryRateDifference
false_discovery_rate_difference = FDRD()
fdrd = false_discovery_rate_difference
(::FDRD)(ft::FairTensor; grp1=:, grp2=:)=abs(false_discovery_rate(ft;grp=grp1)-false_discovery_rate(ft;grp=grp2))

struct FalsePositiveRateRatio <: Measure end
FPRR = FalsePositiveRateRatio
false_positive_rate_ratio = FPRR()
fprr = false_positive_rate_ratio
(::FPRR)(ft::FairTensor; gpr1=:, grp2=:)=false_positve_rate(ft;grp=grp1)/false_positive_rate(ft;grp=grp2)

struct FalseNegativeRateRatio <: Measure end
FNRR = FalseNegativeRateRatio
false_negative_rate_ratio = FNRR()
fnrr = false_negative_rate_ratio
(::FNRR)(ft::FairTensor; gpr1=:, grp2=:)=false_negative_rate(ft;grp=grp1)/false_negative_rate(ft;grp=grp2)


struct FalseOmissionRateRatio <: Measure end
FORR = FalseOmissionRateRatio
false_omission_rate_ratio = FORR()
forr = false_omission_rate_ratio
(::FORR)(ft::FairTensor; grp1=:, grp2=:)=false_omission_rateft;grp=grp1)/false_omission_rate(ft;grp=grp2)

struct FalseDiscoveryRateRatio <: Measure end
FDRR = FalseDiscoveryRateRatio
false_discovery_rate_ratio = FDRR()
fdrr = false_discovery_rate_ratio
(::FDRR)(ft::FairTensor; grp1=:, grp2=:)=false_discovery_rate(ft;grp=grp1)/false_discovery_rate(ft;grp=grp2)

struct AverageOddsDifference <: Measure end
AOD = AverageOddsDifference
average_odds_difference = AOD()
aod = average_odds_difference
(::AOD)(ft::FairTensor; grp1=:; grp2=:)=0.5*(false_positive_rate_difference(ft;grp1=grp1;grp2=grp2)+true_positive_rate_difference(ft;grp1=grp1,grp2=grp2))



