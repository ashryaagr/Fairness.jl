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

It is a postprocessing algorithm that uses JuMP and Ipopt library to minimise error and satisfy the equality of specified specified measures for all groups at the same time.
Automatic differentiation and gradient based optimisation is used to find probabilities with which the predictions are changed for each group.
"""
mutable struct LinProgWrapper{M<:MLJBase.Model} <: DeterministicComposite
	grp::Symbol
	classifier::M
	measure::Measure
end

"""
    LinProgWrapper(classifier=nothing, grp=:class, measure)

Instantiates LinProgWrapper which wraps the classifier and containts the measure to optimised and the sensitive attribute(grp)
"""
function LinProgWrapper(; classifier::MLJBase.Model=nothing, grp::Symbol=:class, measure::Measure)
	model = LinProgWrapper(grp, classifier, measure)
	message = MLJBase.clean!(model)
	isempty(message) || @warn message
	return model
end

function MLJBase.clean!(model::LinProgWrapper)
    warning = ""
	model.classifier!=nothing || (warning *= "No classifier specified in model\n")
    target_scitype(model) <: AbstractVector{<:Finite} || (warning *= "Only Binary Classifiers are supported\n")
    return warning
end

function MMI.fit(model::LinProgWrapper, verbosity::Int, X, y)
	grps = X[:, model.grp]
	n = length(levels(grps)) # Number of different values for sensitive attribute

	# As LinProgWrapper is a postprocessing algorithm, the model needs to be fitted first
	mch = machine(model.classifier, X, y)
	fit!(mch)
	ŷ = MMI.predict(mch, X)

	if typeof(ŷ[1]) <: MLJBase.UnivariateFinite
		ŷ = MLJBase.mode.(ŷ)
	end

	labels = levels(y)
	favLabel = labels[2]
	unfavLabel = labels[1]
	y = y.==favLabel
	ŷ = ŷ.==favLabel

	# Finding the probabilities of changing predictions is a Linear Programming Problem
	# JuMP and Ipopt Optimizer are used to for this Linear Programming Problem
	m = JuMP.Model(Ipopt.Optimizer)

	@variable(m, 0<= p2p[1:n] <=1)
	@variable(m, 0<= p2n[1:n] <=1)
	@variable(m, 0<= n2p[1:n] <=1)
	@variable(m, 0<= n2n[1:n] <=1)

	@constraint(m, [i=1:n], p2p[i] == 1 - p2n[i])
	@constraint(m, [i=1:n], n2p[i] == 1 - n2n[i])

	ft = fair_tensor(categorical(ŷ), categorical(y), categorical(grps))

	vals = zeros(Union{VariableRef, Int, GenericAffExpr{Float64,VariableRef}}, n, 2)
	vals[: , 1] = p2n
	vals[: , 2] = n2p

	_fairTensorLinProg!(ft, vals)

	mat = reshape(ft.mat, (4n))
	@variable(m, aux[1:4n])
	@constraint(m,[i=1:4n], mat[i]==aux[i])

	register(m, :fpr, 4n, (x...)->fpr(Fairness.FairTensor{n}(reshape(collect(x), (n, 2, 2)), ft.labels)), autodiff=true)
	register(m, :fnr, 4n, (x...)->fnr(Fairness.FairTensor{n}(reshape(collect(x), (n, 2, 2)), ft.labels)), autodiff=true)
	@NLobjective(m, Min, fpr(aux...) + fnr(aux...))

	measure = model.measure
	register(m, :func1, 4n, (x...)->measure(Fairness.FairTensor{n}(reshape(collect(x), (n, 2, 2)), ft.labels), grp=levels(grps)[1]), autodiff=true)
	for i in 2:n
		fn_symbol = Symbol("func$i")
		register(m, fn_symbol, 4n, (x...)->measure(Fairness.FairTensor{n}(reshape(collect(x), (n, 2, 2)), ft.labels), grp=levels(grps)[i]), autodiff=true)
		JuMP.add_NL_constraint(m, :($(Expr(:call, fn_symbol, aux...))==$(Expr(:call, Symbol("func1"), aux...))))
		# TODO: Replace call to func1 with a pre-computed expression
	end
	optimize!(m)

	# fitresult will provide the info we require in the predict function.
	fitresult = [[JuMP.value.(p2n), JuMP.value.(n2p)], mch.fitresult, labels]
	# Note: It was necessary to return the levels(y) value in fitreult because in predict there
	# is no way to infer the 2 possible values of labels/targets.
	# Main reason to return values is the edge case : Maybe all of the ŷ predictions are same and we don't get to know both labels

	return fitresult, nothing, nothing
end

# Corresponds to eq_odds function which uses mix_rates to modify results
function MMI.predict(model::LinProgWrapper, fitresult, Xnew)

	(p2n, n2p), classifier_fitresult, labels = fitresult

	ŷ = MMI.predict(model.classifier, classifier_fitresult, Xnew)

	if typeof(ŷ[1]) <: MLJBase.UnivariateFinite
		ŷ = MLJBase.mode.(ŷ)
	end

	favLabel = labels[2]
	unfavLabel = labels[1]

	grps = Xnew[:, model.grp]

	n = length(levels(grps)) # Number of different values for sensitive attribute

	for i in 1:n
		Class = levels(grps)[i]
		Grp = grps .== Class

		pp_indices = shuffle(findall((grps.==Grp) .& (ŷ.==favLabel))) # predicted positive for iᵗʰ class
		pn_indices = shuffle(findall((grps.==Class) .& (ŷ.==unfavLabel))) # predicted negative for iᵗʰ class

		# Note : arrays in julia start from 1
		p2n_indices = pp_indices[1:convert(Int, floor(length(pp_indices)*p2n[i]))]
		n2p_indices = pn_indices[1:convert(Int, floor(length(pn_indices)*n2p[i]))]

		ŷ[p2n_indices] .= unfavLabel
		ŷ[n2p_indices] .= favLabel
	end
	return ŷ
end

MMI.input_scitype(::Type{<:LinProgWrapper{M}}) where M = input_scitype(M)
MMI.target_scitype(::Type{<:LinProgWrapper{M}}) where M = AbstractVector{<:Finite{2}}
