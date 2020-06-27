"""
    EqOddsWrapper

It is a postprocessing algorithm which uses Linear Programming to optimise the constraints for Equalized Odds.
"""
mutable struct EqOddsWrapper <: DeterministicNetwork
	grp::Symbol
	classifier::MLJBase.Model
end

"""
    EqOddsWrapper

Instantiates EqOddsWrapper which wraps the classifier
"""
function EqOddsWrapper(classifier::MLJBase.Model; grp::Symbol=:class)
	return EqOddsWrapper(grp, classifier)
end

# Corresponds to eq_odds_optimal_mix_rates function, mix_rates are returned as fitresult
function MMI.fit(model::EqOddsWrapper, verbosity::Int,
	X, y)
	grps = X[:, model.grp]
	length(levels(grps))==2 || throw(ArgumentError("This algorithm supports only groups with 2 different values only"))

	# As equalized odds is a postprocessing algorithm, the model needs to be fitted first
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

	# The prefix s Corresponds to priveledged class
	@variable(m, 0<= sp2p <=1)
	@variable(m, 0<= sp2n <=1)
	@variable(m, 0<= sn2p <=1)
	@variable(m, 0<= sn2n <=1)

	# The prefix o Corresponds to unpriveledged class
	@variable(m, 0<= op2p <=1)
	@variable(m, 0<= op2n <=1)
	@variable(m, 0<= on2p <=1)
	@variable(m, 0<= on2n <=1)

	@constraint(m, constraint1, sp2p == 1 - sp2n)
	@constraint(m, constraint2, sn2p == 1 - sn2n)
	@constraint(m, constraint3, op2p == 1 - op2n)
	@constraint(m, constraint4, on2p == 1 - on2n)

	privClass = levels(grps)[2]
	unprivClass = levels(grps)[1]
	priv = grps .== privClass
	unpriv = grps .== unprivClass

	sflip = 1 .- ŷ[priv]
	sconst = ŷ[priv]
	oflip = 1 .- ŷ[unpriv]
	oconst = ŷ[unpriv]

	sbr = mean(y[priv]) # Base rate for priviledged class
	obr = mean(y[unpriv]) # Base rate for unpriveledged class

	ft = fair_tensor(categorical(ŷ), categorical(y), categorical(grps))
	sfpr = fpr(ft; grp=privClass) * sp2p + tnr(ft; grp=privClass) * sn2p
	sfnr = fnr(ft; grp=privClass) * sn2n + tpr(ft; grp=privClass) * sp2n
	ofpr = fpr(ft; grp=unprivClass) * op2p + tnr(ft; grp=unprivClass) * on2p
	ofnr = fnr(ft; grp=unprivClass) * on2n + tpr(ft; grp=unprivClass) * op2n
	error = sfpr + sfnr + ofpr + ofnr
	@objective(m, Min, error)

	sm_tn = ŷ[priv].==0 .& y[priv].==0
	sm_fn = ŷ[priv].==0 .& y[priv].==1
	sm_tp = ŷ[priv].==1 .& y[priv].==1
	sm_fp = ŷ[priv].==1 .& y[priv].==0

	om_tn = ŷ[unpriv].==0 .& y[unpriv].==0
	om_fn = ŷ[unpriv].==0 .& y[unpriv].==1
	om_tp = ŷ[unpriv].==1 .& y[unpriv].==1
	om_fp = ŷ[unpriv].==1 .& y[unpriv].==0

	# Following variables names have been changed from the implementation by Equalized Odds postprocessing algorithm
	# These variables better explain the corresponding quantity.
	# For eg. spp_given_p Corresponds to Predicted Positive given Negative for priveledged class
	spp_given_p = ((sn2p * mean(sflip .& sm_fn) + sn2n * mean(sconst .& sm_fn)) / sbr +
				  (sp2p * mean(sconst .& sm_tp) + sp2n * mean(sflip .& sm_tp)) / sbr)

	spn_given_n = ((sp2n * mean(sflip .& sm_fp) + sp2p * mean(sconst .& sm_fp)) / (1 - sbr) +
				  (sn2p * mean(sflip .& sm_tn) + sn2n * mean(sconst .& sm_tn)) / (1 - sbr))

	opp_given_p = ((on2p * mean(oflip .& om_fn) + on2n * mean(oconst .& om_fn)) / obr +
				  (op2p * mean(oconst .& om_tp) + op2n * mean(oflip .& om_tp)) / obr)

	opn_given_n = ((op2n * mean(oflip .& om_fp) + op2p * mean(oconst .& om_fp)) / (1 - obr) +
				  (on2p * mean(oflip .& om_tn) + on2n * mean(oconst .& om_tn)) / (1 - obr))

	@constraint(m, constraint5, spp_given_p==opp_given_p)
	@constraint(m, constraint6, spn_given_n==opn_given_n)

	optimize!(m)

	fitresult = [[JuMP.value(sp2n), JuMP.value(sn2p), JuMP.value(op2n), JuMP.value(on2p)], mch.fitresult]

	return fitresult, nothing, nothing
end

# Corresponds to eq_odds function which uses mix_rates to modify results
function MMI.predict(model::EqOddsWrapper, fitresult, Xnew)

	(sp2n, sn2p, op2n, on2p), classifier_fitresult = fitresult

	ŷ = MMI.predict(model.classifier, classifier_fitresult, Xnew)

	if typeof(ŷ[1]) <: MLJBase.UnivariateFinite
		ŷ = MLJBase.mode.(ŷ)
	end

	ŷ = convert(Array, ŷ) # Need to convert to normal array as categorical array doesn't support sub
	grps = Xnew[:, model.grp]

	privClass = levels(grps)[2]
	unprivClass = levels(grps)[1]
	priv = grps .== privClass
	unpriv = grps .== unprivClass


	s_pp_indices = shuffle(findall((grps.==privClass) .& (ŷ.==1))) # predicted positive for priv class
	s_pn_indices = shuffle(findall((grps.==privClass) .& (ŷ.==0))) # predicted negative for unpriv class
	o_pp_indices = shuffle(findall((grps.==unprivClass) .& (ŷ.==1))) # predicted positive for priv class
	o_pn_indices = shuffle(findall((grps.==unprivClass) .& (ŷ.==0))) # predicted negative for unpriv class

	# Note : arrays in julia start from 1
	s_p2n_indices = s_pp_indices[1:convert(Int, floor(length(s_pp_indices)*sp2n))]
	s_n2p_indices = s_pn_indices[1:convert(Int, floor(length(s_pn_indices)*sn2p))]
	o_p2n_indices = o_pp_indices[1:convert(Int, floor(length(o_pp_indices)*op2n))]
	o_n2p_indices = o_pn_indices[1:convert(Int, floor(length(o_pn_indices)*on2p))]

	ŷ[s_p2n_indices] = 1 .- ŷ[s_p2n_indices]
	ŷ[s_n2p_indices] = 1 .- ŷ[s_n2p_indices]
	ŷ[o_p2n_indices] = 1 .- ŷ[o_p2n_indices]
	ŷ[o_n2p_indices] = 1 .- ŷ[o_n2p_indices]

	return ŷ
end
