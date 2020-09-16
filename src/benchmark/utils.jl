function disparate_impact(ft::Fairness.FairTensor; priv_grps::Array{String, 1})
	num, den = 0, 0
	priv_count, unpriv_count = 0, 0
	for grp in ft.labels
		if grp in priv_grps
			den += sum(ft[Fairness._ftIdx(ft, grp), 1, :])
			priv_count += sum(ft[Fairness._ftIdx(ft, grp), :, :])
		else
			num += sum(ft[Fairness._ftIdx(ft, grp), 1, :])
			unpriv_count += sum(ft[Fairness._ftIdx(ft, grp), :, :])
		end
	end
	return (num/unpriv_count)/(den/priv_count)
end

struct DisparateImpact <: Fairness.Measure
	grp::Symbol
	priv_grps::Array{String, 1}
end

function DisparateImpact(;grp, priv_grps)
	return DisparateImpact(grp, priv_grps)
end

function (di::DisparateImpact)(ŷ, X, y)
	if typeof(ŷ[1]) <: MLJBase.UnivariateFinite
		ŷ = StatsBase.mode.(ŷ)
	end
	grps = X[:, di.grp]
	ft = fair_tensor(categorical(ŷ), categorical(y), categorical(grps))
	return disparate_impact(ft, priv_grps=di.priv_grps)
end

function (di::DisparateImpact)(ft::FairTensor; grp=:)
	num, den = 0, 0
	priv_count, unpriv_count = 0, 0
	priv_grps = di.priv_grps
	for grp in priv_grps
		den += sum(_calcmetric(ft, grp, 1, :))
		priv_count += sum(ft[Fairness._ftIdx(ft, grp), :, :])
	end
	num += sum(_calcmetric(ft, grp, 1, :))
	unpriv_count += sum(_calcmetric(ft, grp, :, :))
	return (num/unpriv_count)/(den/priv_count)
end

MLJBase.name(::Type{<:DisparateImpact}) = "DisparateImpact"
# MLJBase.target_scitype(::Type{<:DisparateImpact}) = AbstractArray{Multiclass{2},1}
MLJBase.supports_weights(::Type{<:DisparateImpact}) = false # for now
# MLJBase.prediction_type(::Type{<:DisparateImpact}) = :deterministic # Not specifying it to have check_measures false
MLJBase.orientation(::Type{<:DisparateImpact}) = :other # other options are :score, :loss
MLJBase.reports_each_observation(::Type{<:DisparateImpact}) = false
MLJBase.aggregation(::Type{<:DisparateImpact}) = Fairness.Mean()
MLJBase.is_feature_dependent(::Type{<:DisparateImpact}) = true
