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
	score(x) = x.prob_given_ref[2]
	all_indices = convert(Array, 1:n)
	for itr in 1:n_model.steps
		y_train = shuffle!(all_indices)[1:Int(round(0.7n))]
		y_tune = a[Int(round(0.7n))+1:n]
		mch = machine(model.classifier, X, y)
		fit!(mch)
		ŷ = MMI.predict(mch, X)
		for i in 1:n_grps
			Xnew = MLJBase.transform(MLJBase.fit!(machine(ContinuousEncoder(), X)), X)
			Xnew.y = convert(Array, y)
			Xnew.scores = score.(ŷ)
			i=1
			indices = grps.==levels_[i]
			scores = score.(ŷ[indices])
			dis = Distributions.fit(Distributions.Normal, scores)
			md = Distributions.MvNormal(StatsBase.mean.(eachcol(Xnew)), StatsBase.std.(eachcol(Xnew)))
			ft = FairTensor(fill(0.25, (n_grps, 2, 2)), levels(grps))
			for j in 1:length(scores)
				a, b, c, d = [convert(Array, Xnew[j, names(Xnew)[1:length(names(Xnew))-2]]) for _ in 1:4]
				pr = zeros(2, 2)
				# TODO: The scores will hardly ever be 1. So, pdf calculations below are wrong!
				# We should instead use λ to calculate the values in fairness metric.
				pr[1, 1] = pdf(md, push!(a, 1, 1)) #TruePositive
				pr[1, 2] = pdf(md, push!(b, 0, 1)) #FalsePositive
				pr[2, 1] = pdf(md, push!(c, 1, 0)) #FalseNegative
				pr[2, 2] = pdf(md, push!(d, 0, 0)) #TrueNegative
				pr /= sum(pr)
				ft[i, :, :] = pr
			end
			loss(λ) = mean(accuracy(ft)) + model.alpha*measure(ft)^2

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
