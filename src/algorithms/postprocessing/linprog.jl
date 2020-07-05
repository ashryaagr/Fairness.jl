Base.zero(::Type{Union{Int64, VariableRef, GenericAffExpr{Float64,VariableRef}}}) = 0

# Helper function to modify the fairness tensor according to the values for sp2n, on2p, etc
# vals is a 2D array of the form [[sp2n, sn2p], [op2n, on2p]]
function _fairTensorLinProg!(ft::FairTensor, vals)
	mat = deepcopy(ft.mat)
	ft.mat = zeros(Union{VariableRef, Int, GenericAffExpr{Float64,VariableRef}}, size(mat)...)
	for i in 1:length(ft.labels)
		p2n, n2p = vals[i, :] #These vals are VariableRef from library JuMP
		p2p, n2n = 1-p2n, 1-n2p
		a = zeros(Union{VariableRef, Int, GenericAffExpr{Float64,VariableRef}}, 2, 2) # The numbers for modified fairness tensor values for a group
		a[1, 1] = mat[i, 1, 1]*p2p + mat[i, 2, 2]*n2p
		a[1, 2] = mat[i, 1, 2]*p2p + mat[i, 2, 1]*n2p
		a[2, 1] = mat[i, 2, 1]*n2n + mat[i, 1, 1]*p2n
		a[2, 2] = mat[i, 2, 2]*n2n + mat[i, 1, 2]*p2n
		ft.mat[i, :, :] = a
	end
end

"""
    LinProgWrapper

It is a postprocessing algorithm that uses JuMP and Ipopt library to minimise error and satisfy the specified constraint at the same time.
Automatic differentiation and gradient based optimisation is used to find probabilities with which the predictions are changed for each group.
"""
mutable struct LinProgWrapper <: DeterministicNetwork
	grp::Symbol
	classifier::MLJBase.Model
	measure::Measure
end

"""
    LinProgWrapper(classifier; grp=:class, measure)

Instantiates LinProgWrapper which wraps the classifier and containts the measure to optimised and the sensitive attribute(grp)
"""
function LinProgWrapper(classifier::MLJBase.Model; grp::Symbol=:class, measure::Measure)
	return LinProgWrapper(grp, classifier, measure)
end

function MMI.fit(model::LinProgWrapper, verbosity::Int, X, y)
	grps = X[:, model.grp]
	length(levels(grps))==2 || throw(ArgumentError("This algorithm supports only groups with 2 different values only"))

	# As LinProgWrapper is a postprocessing algorithm, the model needs to be fitted first
	mch = machine(model.classifier, X, y)
	fit!(mch)
	ŷ = MMI.predict(mch, X)

	if typeof(ŷ[1]) <: MLJBase.UnivariateFinite
		ŷ = MLJBase.mode.(ŷ)
	end

	ŷ = convert(Array, ŷ) # Incase ŷ is categorical array, convert to normal array to support various operations
	y = convert(Array, y)

	# Finding the probabilities of changing predictions is a Linear Programming Problem
	# JuMP and Ipopt Optimizer are used to for this Linear Programming Problem
	m = JuMP.Model(Ipopt.Optimizer)

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

	ft = fair_tensor(categorical(ŷ), categorical(y), categorical(grps))

	vals = zeros(Union{VariableRef, Int, GenericAffExpr{Float64,VariableRef}}, 2, 2)
	vals[1, :] = [sp2n, sn2p]
	vals[2, :] = [op2n, on2p]

	_fairTensorLinProg!(ft, vals)

	mat = reshape(ft.mat, (8))
	@variable(m, aux[1:8])
	@constraint(m, cons[i=1:8], mat[i]==aux[i])

	register(m, :fpr, 8, (x...)->fpr(MLJFair.FairTensor{2}(reshape(collect(x), (2, 2, 2)), ft.labels)), autodiff=true)
	register(m, :fnr, 8, (x...)->fnr(MLJFair.FairTensor{2}(reshape(collect(x), (2, 2, 2)), ft.labels)), autodiff=true)
	@NLobjective(m, Min, fpr(aux...) + fnr(aux...))

	measure = model.measure
	register(m, :ms1, 8, (x...)->measure(MLJFair.FairTensor{2}(reshape(collect(x), (2, 2, 2)), ft.labels), grp=levels(grps)[1]), autodiff=true)
	register(m, :ms2, 8, (x...)->measure(MLJFair.FairTensor{2}(reshape(collect(x), (2, 2, 2)), ft.labels), grp=levels(grps)[2]), autodiff=true)

	@NLexpression(m, m1, ms1(aux...))
	@NLexpression(m, m2, ms2(aux...))
	@NLconstraint(m, constraint5, m1==m2)

	optimize!(m)

	fitresult = [[JuMP.value(sp2n), JuMP.value(sn2p), JuMP.value(op2n), JuMP.value(on2p)], mch.fitresult]

	return fitresult, nothing, nothing
end

# Corresponds to eq_odds function which uses mix_rates to modify results
function MMI.predict(model::LinProgWrapper, fitresult, Xnew)

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
