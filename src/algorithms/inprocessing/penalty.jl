"""
	PenaltyWrapper

It is an inprocessing algorithm that wraps a Probabilistic classifier.
Optimal thresholds for each group in protected attribute is found to minimise accuracy+ alpha*fairness_measure^2.
Gradients are used to find optimal threshold values. alpha controls the fairness accuracy tradeoff.
"""
struct PenaltyWrapper{M<:MLJBase.Model} <: DeterministicComposite
	grp::Symbol
	classifier::M
	measure::MLJBase.Measure
	alpha::Real
	n_iters::Int
	lr::Float64
end

"""
	PenaltyWrapper(classifier=nothing, grp=:class, measure, alpha=1, n_iters=1000, lr=0.01)

Instantiates PenaltyWrapper which wraps the classifier and contains measure to be optimised.
It also provides various hyperparameters to the wrapper.
"""
function PenaltyWrapper(;
	grp::Symbol = :class,
    classifier::MLJBase.Model = nothing,
    measure::MLJBase.Measure,
    alpha::Real = 1, # Fairess-Accuracy tradeoff parameter in loss function
    n_iters = 10^3, # Number of gradient steps to perform per group
	lr::Float64 = 10^-2 # Learning Rate
)
    model = PenaltyWrapper(grp, classifier, measure, alpha, n_iters, lr)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJBase.clean!(model::PenaltyWrapper)
	warning = ""
    model.classifier!=nothing || (warning *= "No classifier specified in model\n")
    target_scitype(model.classifier) <: AbstractVector{<:Finite} || (warning *= "Only Binary Classifiers are supported\n")
    return warning
end

function MMI.fit(model::PenaltyWrapper, verbosity::Int,
	X, y)
	grps = X[:, model.grp]
	labels = levels(y)
	favLabel = labels[2]
	unfavLabel = labels[1]
	n_grps = length(levels(grps)) # Number of different values for sensitive attribute
	n = length(y)

	y = categorical(y.==favLabel)

	grp_idx = Dict()
	levels_ = levels(grps)
	for i in 1:n_grps
	    grp_idx[levels_[i]] = i
	end

	λ = fill(0.5, n_grps) #Thresholds for various groups
	mat = Array{Any, 3}(undef, n_grps, 2, 2)
	mat .= 0.25
	ft = FairTensor(mat, levels(grps))

	score(x) = x.prob_given_ref[2]
	all_indices = convert(Array, 1:n)
	for itr in 1:model.n_iters
		train_indices = shuffle!(all_indices)[1:Int(round(0.7n))]
		tune_indices = all_indices[Int(round(0.7n))+1:n]
		mch = machine(model.classifier, X[train_indices, :], y[train_indices])
		fit!(mch, verbosity=0)
		ŷ = MMI.predict(mch, X[tune_indices, :])
		μ = mean(score.(ŷ))
		pr_max = maximum(score.(ŷ))
		pr_min = minimum(score.(ŷ))
		for i in 1:n_grps
			indices = grps[tune_indices].==levels_[i]
			y_tune_indices = convert(Array, y[tune_indices][indices])
			if all(y_tune_indices.==unfavLabel) || all(y_tune_indices.==favLabel) continue end
			scores = score.(ŷ[indices])
			dis0 = Distributions.fit(Distributions.Normal, scores[y_tune_indices .== unfavLabel])
			dis1 = Distributions.fit(Distributions.Normal, scores[y_tune_indices .== favLabel])
			function lambdafair(λ)
				pr = Array{Any, 2}(undef, 2, 2)
				pr[1, 1] = 1-Distributions.cdf(dis1, λ) #TruePositive
				pr[1, 2] = 1-Distributions.cdf(dis0, λ) #FalsePositive
				pr[2, 1] = Distributions.cdf(dis1, λ) #FalseNegative
				pr[2, 2] = Distributions.cdf(dis0, λ) #TrueNegative
				pr /= sum(pr)
				ft[i, :, :] = pr
				ft
			end
			compositeloss(ft::FairTensor) =  accuracy(ft) +
				model.alpha*(model.measure(ft, grp=levels_[i])-model.measure(ft, grp=levels_[1]))^2
			loss(λ) = compositeloss(lambdafair(λ[1])) + ((λ[1]-μ))^2/(pr_max-pr_min)

			λᵢ_grad = ForwardDiff.gradient(loss, [λ[i]])
			λ[i] -= λᵢ_grad[1]*model.lr
		end
	end

	# Make y and ŷ as an attribute. And then predict whether that is true or false
	mach = machine(model.classifier, X, y)
	fit!(mach, verbosity=verbosity)
	fitresult = [mach.fitresult, λ]
	return fitresult, nothing, nothing
end

function MMI.predict(model::PenaltyWrapper, fitresult, Xnew)
	classifier_fitresult, λ = fitresult
	n = size(Xnew)[1]
	grps = Xnew[:, model.grp]
	levels_ = levels(grps)
	n_grps = length(levels(grps))
	grp_idx = Dict() # Maps group_name => index of group in levels_
	for i in 1:n_grps
		grp_idx[levels_[i]] = i
	end

	score(x) = x.prob_given_ref[2]
	preds = MMI.predict(model.classifier, classifier_fitresult, Xnew) # preds are UnivariateFinite
	scores = score.(preds)
	ŷ = zeros(Int, n)
	for i in 1:n
		ŷ[i] = scores[i]>λ[grp_idx[Xnew[i, model.grp]]]
	end
	return ŷ
end

MMI.input_scitype(::Type{<:PenaltyWrapper{M}}) where M = input_scitype(M)
MMI.target_scitype(::Type{<:PenaltyWrapper{M}}) where M = AbstractVector{<:Finite{2}}
