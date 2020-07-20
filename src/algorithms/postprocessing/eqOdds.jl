"""
    EqOddsWrapper

It is a postprocessing algorithm which uses Linear Programming to optimise the constraints for Equalized Odds.
"""
mutable struct EqOddsWrapper{M<:MLJBase.Model} <: DeterministicComposite
	grp::Symbol
	classifier::M
end

"""
    EqOddsWrapper(classifier=nothing, grp=:class)

Instantiates EqOddsWrapper which wraps the classifier
"""
function EqOddsWrapper(; classifier::MLJBase.Model=nothing, grp::Symbol=:class)
	return EqOddsWrapper(grp, classifier)
end

function MLJBase.clean!(model::EqOddsWrapper)
    warning = ""
	model.classifier!=nothing || (warning *= "No classifier specified in model")
    target_scitype(model) <: AbstractVector{<:Finite} || (warning *= "Only Binary Classifiers are supported")
    return warning
end

# Corresponds to eq_odds_optimal_mix_rates function, mix_rates are returned as fitresult
function MMI.fit(model::EqOddsWrapper, verbosity::Int,
	X, y)
	grps = X[:, model.grp]
	n = length(levels(grps)) # Number of different values for sensitive attribute

	# As equalized odds is a postprocessing algorithm, the model needs to be fitted a_
	mch = machine(model.classifier, X, y)
	fit!(mch)
	ŷ = MMI.predict(mch, X)

	if typeof(ŷ[1]) <: MLJBase.UnivariateFinite
		ŷ = MLJBase.mode.(ŷ)
	end

	ŷ = convert(Array, ŷ) # Incase ŷ is categorical array, convert to normal array to support various operations
	y = convert(Array, y)

	# Finding the probabilities of changing predictions is a Linear Programming Problem
	# JuMP and Cbc are used to for this Linear Programming Problem
	m = JuMP.Model(GLPK.Optimizer)

	@variable(m, 0<= p2p[1:n] <=1)
	@variable(m, 0<= p2n[1:n] <=1)
	@variable(m, 0<= n2p[1:n] <=1)
	@variable(m, 0<= n2n[1:n] <=1)

	@constraint(m, cons1[i=1:n], p2p[i] == 1 - p2n[i])
	@constraint(m, cons2[i=1:n], n2p[i] == 1 - n2n[i])

	a_Class = levels(grps)[1]
	a_Grp = grps .== a_Class

	a_flip = 1 .- ŷ[a_Grp]
	a_const = ŷ[a_Grp]

	a_BaseRate = mean(y[a_Grp]) # Base rate for a_ class

	ft = fair_tensor(categorical(ŷ), categorical(y), categorical(grps))

	error = 0
	for i in 1:n
		i_fpr = fpr(ft; grp=levels(grps)[i]) * p2p[i] + tnr(ft; grp=levels(grps)[i]) * n2p[i]
		i_fnr = fnr(ft; grp=levels(grps)[i]) * n2n[i] + tpr(ft; grp=levels(grps)[i]) * p2n[i]
		error += (i_fpr + i_fnr)
	end
	@objective(m, Min, error)

	a_tn = ŷ[a_Grp].==0 .& y[a_Grp].==0
	a_fn = ŷ[a_Grp].==0 .& y[a_Grp].==1
	a_tp = ŷ[a_Grp].==1 .& y[a_Grp].==1
	a_fp = ŷ[a_Grp].==1 .& y[a_Grp].==0

	# Following variables names have been changed from the implementation by Equalized Odds postprocessing algorithm
	# These variables better explain the corresponding quantity.
	# For eg. a_pp_given_p Corresponds to Predicted Positive given Negative for class named a

	a_pp_given_p = ((n2p[1] * mean(a_flip .& a_fn) + n2n[1] * mean(a_const .& a_fn)) / a_BaseRate +
				  (p2p[1] * mean(a_const .& a_tp) + p2n[1] * mean(a_flip .& a_tp)) / a_BaseRate)

	a_pn_given_n = ((p2n[1] * mean(a_flip .& a_fp) + p2p[1] * mean(a_const .& a_fp)) / (1 - a_BaseRate) +
				  (n2p[1] * mean(a_flip .& a_tn) + n2n[1] * mean(a_const .& a_tn)) / (1 - a_BaseRate))

	for i in 2:n
		i_Class = levels(grps)[i]

		i_Grp = grps .== i_Class

		i_flip = 1 .- ŷ[i_Grp]
		i_const = ŷ[i_Grp]

		i_BaseRate = mean(y[i_Grp]) # Base rate for a_ class

		i_tn = ŷ[i_Grp].==0 .& y[i_Grp].==0
		i_fn = ŷ[i_Grp].==0 .& y[i_Grp].==1
		i_tp = ŷ[i_Grp].==1 .& y[i_Grp].==1
		i_fp = ŷ[i_Grp].==1 .& y[i_Grp].==0

		i_pp_given_p = ((n2p[i] * mean(i_flip .& i_fn) + n2n[i] * mean(i_const .& i_fn)) / i_BaseRate +
					  (p2p[i] * mean(i_const .& i_tp) + p2n[i] * mean(i_flip .& i_tp)) / i_BaseRate)

		i_pn_given_n = ((p2n[i] * mean(i_flip .& i_fp) + p2p[i] * mean(i_const .& i_fp)) / (1 - i_BaseRate) +
					  (n2p[i] * mean(i_flip .& i_tn) + n2n[i] * mean(i_const .& i_tn)) / (1 - i_BaseRate))

		@constraint(m, a_pp_given_p==i_pp_given_p)
		@constraint(m, a_pn_given_n==i_pn_given_n)
	end

	optimize!(m)

	fitresult = [[JuMP.value.(p2n), JuMP.value.(n2p)], mch.fitresult]

	return fitresult, nothing, nothing
end

# Corresponds to eq_odds function which uses mix_rates to modify results
function MMI.predict(model::EqOddsWrapper, fitresult, Xnew)

	(p2n, n2p), classifier_fitresult = fitresult

	ŷ = MMI.predict(model.classifier, classifier_fitresult, Xnew)

	if typeof(ŷ[1]) <: MLJBase.UnivariateFinite
		ŷ = MLJBase.mode.(ŷ)
	end

	ŷ = convert(Array, ŷ) # Need to convert to normal array as categorical array doesn't support sub
	grps = Xnew[:, model.grp]

	n = length(levels(grps)) # Number of different values for sensitive attribute

	for i in 1:n
		Class = levels(grps)[i]
		Grp = grps .== Class

		pp_indices = shuffle(findall((grps.==Grp) .& (ŷ.==1))) # predicted positive for iᵗʰ class
		pn_indices = shuffle(findall((grps.==Class) .& (ŷ.==0))) # predicted negative for iᵗʰ class

		# Note : arrays in julia start from 1
		p2n_indices = pp_indices[1:convert(Int, floor(length(pp_indices)*p2n[i]))]
		n2p_indices = pn_indices[1:convert(Int, floor(length(pn_indices)*n2p[i]))]

		ŷ[p2n_indices] = 1 .- ŷ[p2n_indices]
		ŷ[n2p_indices] = 1 .- ŷ[n2p_indices]
	end
	return ŷ
end

MMI.input_scitype(::Type{<:EqOddsWrapper{M}}) where M = input_scitype(M)
MMI.target_scitype(::Type{<:EqOddsWrapper{M}}) where M = AbstractVector{<:Finite{2}}
