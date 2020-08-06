struct PenaltyWrapper{M<:MLJBase.Model} <: DeterministicComposite
	grp::Symbol
	classifier::M
	measure::MLJBase.Measure
	alpha::Real
	n_iters::Int
	lr::Float64
end

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
	model = PenaltyWrapper(classifier=ConstantClassifier(), grp=:Sex, measure=tpr)
	X, y = @load_toydata
	measure=tpr
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
	ft = FairTensor(fill(0.25, (n_grps, 2, 2)), levels(grps))
	score(x) = x.prob_given_ref[2]
	all_indices = convert(Array, 1:n)
	for itr in 1:n_iters
		train_indices = shuffle!(all_indices)[1:Int(round(0.7n))]
		tune_indices = a[Int(round(0.7n))+1:n]
		mch = machine(model.classifier, X[train_indices, :], y[train_indices])
		fit!(mch)
		ŷ = MMI.predict(mch, X[tune_indices, :])
		for i in 1:n_grps
			Xnew = MLJBase.transform(MLJBase.fit!(machine(ContinuousEncoder(), X[tune_indices])), X[tune_indices])
			Xnew.y = convert(Array, y[tune_indices])
			Xnew.scores = score.(ŷ)
			indices = grps[tune_indices].==levels_[i]
			scores = score.(ŷ[indices])
			dis0 = Distributions.fit(Distributions.Normal, scores[Xnew.y.==unfavLabel])
			dis1 = Distributions.fit(Distributions.Normal, scores[Xnew.y.==favLabel])
			function lambdafair(λ)
				pr = zeros(2, 2)
				pr[1, 1] = cdf(dis1, λ) #TruePositive
				pr[1, 2] = cdf(dis0, λ) #FalsePositive
				pr[2, 1] = cdf(dis1, λ) #FalseNegative
				pr[2, 2] = cdf(dis0, λ) #TrueNegative
				pr /= sum(pr)
				ft[i, :, :] = pr
				ft
			end
			compositeloss(ft::FairTensor) =  mean(accuracy(ft)) + model.alpha*measure(ft)^2
			loss(λ) = compositeloss(lambdafair(λ))

			λᵢ_grad = ForwardDiff.gradient(loss, [0.6])
			λ[i] -= λᵢ_grad*model.lr
		end
	end

	# Make y and ŷ as an attribute. And then predict whether that is true or false
	fitresult = [mch.fitresult, λ]
	return fitresult, nothing, nothing
end

function MMI.predict(model::PenaltyWrapper, fitresult, Xnew)
	classifier_fitresult, λ = fitresult

	n = size(Xnew)[1]
	grps = model.grp
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
		ŷ[i] = scores[i]>λ[grp_idx[X[i, model.grp]]]
	end
	return ŷ
end

MMI.input_scitype(::Type{<:PenaltyWrapper{M}}) where M = input_scitype(M)
MMI.target_scitype(::Type{<:PenaltyWrapper{M}}) where M = AbstractVector{<:Finite{2}}
