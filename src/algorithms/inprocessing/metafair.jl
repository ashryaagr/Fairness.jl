mutable struct MetaFairWrapper{M<:MLJBase.Model} <: DeterministicComposite
	grp::Symbol
	classifier::M
	measure::MLJBase.Measure
end

function MetaFairWrapper(; classifier::MLJBase.Model=nothing, grp::Symbol=:class, measure::MLJBase.Measure)
	model = MetaFairWrapper(grp, classifier, measure)
	message = MLJBase.clean!(model)
	isempty(message) || @warn message
	return model
end

function MLJBase.clean!(model::MetaFairWrapper)
	warning = ""
    model.classifier!=nothing || (warning *= "No classifier specified in model\n")
    target_scitype(model.classifier) <: AbstractVector{<:Finite} || (warning *= "Only Binary Classifiers are supported\n")
    return warning
end

function MMI.fit(model::MetaFairWrapper, verbosity::Int,
	X, y)
	grps = X[:, model.grp]
	n = length(levels(grps)) # Number of different values for sensitive attribute

	m = JuMP.Model(Ipopt.Optimizer)


	mch = machine(model.classifier, X, y)
	fit!(mch)
	fitresult = []
	return fitresult, nothing, nothing
end

function MMI.predict(model::MetaFairWrapper, fitresult, Xnew)

end

MMI.input_scitype(::Type{<:MetaFairWrapper{M}}) where M = input_scitype(M)
MMI.target_scitype(::Type{<:MetaFairWrapper{M}}) where M = AbstractVector{<:Finite{2}}
