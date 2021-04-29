
"""
        CalEqOddsWrapper
Calibrated equalized odds is a post-processing technique that optimizes over calibrated classifier score outputs to find probabilities with which to change output labels with an equalized odds objective

"""
	
struct CalEqOddsWrapper{M<:MLJBase.Model} <: Deterministic
	grp::Symbol
        classifier::M
	fp_rate::Int64
	fn_rate::Int64
	alpha::Float64
end
	
"""
        CalEqOddsWrapper(classifier=nothing, grp=:class, fp_rate=1, fn_rate=1)
Instantiates CalEqOddsWrapper which wraps the classifier
"""
function CalEqOddsWrapper(; classifier::MLJBase.Model=nothing,grp::Symbol=:class, fp_rate=1, fn_rate=1, alpha=1.0)
        model =  CalEqOddsWrapper(grp, classifier, fp_rate, fn_rate, alpha)
	message = MLJBase.clean!(model)
	isempty(message) || @warn message
	return model
end
	
function MLJBase.clean!(model::CalEqOddsWrapper)
	warning = ""
	model.classifier!=nothing || (warning *= "No classifier specified in model\n")
	target_scitype(model) <: AbstractVector{<:Finite} || (warning *= "Only Binary Classifiers are supported\n")
	(model.alpha>=0 && model.alpha<=1) || (warning*="alpha should be between 0 and 1 (inclusive)\n")
	return warning
end
	
	# returns mixrates
function MMI.fit(model::CalEqOddsWrapper, verbosity::Int, X, y)
	grps = X[:, model.grp]
	n = length(levels(grps)) # Number of different values for sensitive attribute
	# As equalized odds is a postprocessing algorithm, the model needs to be fitted a_
	fp_rate = model.fp_rate
	fn_rate = model.fn_rate
	mch = machine(model.classifier, X, y)
	fit!(mch)
	ŷ = MMI.predict(mch, X)
	if typeof(ŷ[1]) <: MLJBase.UnivariateFinite
		ŷ = MLJBase.mode.(ŷ)
	end
	ft = fair_tensor(ŷ, y, grps)
	labels = levels(y)
	favLabel = labels[2]
	unfavLabel = labels[1]
	y = y.==favLabel
	ŷ = ŷ.==favLabel
	# Compares the unprivileged class against all others
	g = levels(grps)
	c = grps.==g[sortperm(ft[:,fp_rate*1+fn_rate*2,fp_rate*2+fn_rate*1])[1]]
	for i in 1:Int8(length(sortperm(ft[:,fp_rate*1+fn_rate*2,fp_rate*2+fn_rate*1]))/2)
		a = grps.==g[i]
		c = a.|c
	end
		
	a_Class = levels(grps)[1]
	a_Grp = c
	b_Grp = .~c
	b_const = ŷ[b_Grp]
	a_const = ŷ[a_Grp]
	br1 = mean(y[a_Grp])
	br2 = mean(y[b_Grp])
	mix_rate1,mix_rate2 = CalEqOdds(ŷ[a_Grp],ŷ[b_Grp],y[a_Grp],y[b_Grp],fp_rate,fn_rate)
	
	fitresult = [mix_rate1, mix_rate2, br1, br2, mch.fitresult, labels]
	return fitresult, nothing, nothing
end
	
# Modifies the predictions of the model based on the mixrates
function MMI.predict(model::CalEqOddsWrapper, fitresult, Xnew)
	mix_rate1, mix_rate2, y1br, y2br,classifier_fitresult, labels = fitresult
	ŷ = MMI.predict(model.classifier, classifier_fitresult, Xnew)
	if typeof(ŷ[1]) <: MLJBase.UnivariateFinite
		ŷ = MLJBase.mode.(ŷ)
	end
	favLabel = labels[2]
	unfavLabel = labels[1]
	
	grps = Xnew[:, model.grp]
	g = levels(grps)
	c = grps.==g[sortperm(ft[:,2,2])[1]]
	for i in 1:Int8(length(sortperm(ft[:,2,2]))/2)
		a = grps.==g[i]
		c = a.|c
	end
	a_Grp = c
	b_Grp = .~c
	a_Class = levels(grps)[1]
	a_const = shuffle(findall(c))
	b_const = shuffle(findall(.~c))
	indices1 = a_const[1:convert(Int64, floor((mix_rate1)*(length(a_const))))]
	indices2 = b_const[1:convert(Int64, floor((mix_rate2)*(length(b_const))))]
	new_ŷ = deepcopy(ŷ)
	new_ŷ[indices1] .= convert(Int64,round(y1br))
	new_ŷ[indices2] .= convert(Int64,round(y2br))
	return new_ŷ
end
	
MMI.input_scitype(::Type{<:CalEqOddsWrapper{M}}) where M = input_scitype(M)
MMI.target_scitype(::Type{<:CalEqOddsWrapper{M}}) where M = AbstractVector{<:Finite{2}}
	
# Corresponds to calib_eq_odds function which calculates mixrates
function CalEqOdds(ŷ1, ŷ2, y1, y2, fp_rate, fn_rate)
	y1_triv = ones(length(y1))*mean(y1)
	y2_triv = ones(length(y2))*mean(y2)
	if fn_rate == 0
		cost1 = mean(ŷ1[y1.==0])
		cost2 = mean(ŷ2[y2.==0])
		trivial_cost1 = mean(y1_triv[y1.==0])
		trivial_cost2 = mean(y2_triv[y2.==0])
	elseif fp_rate == 0
		cost1 = 1-mean(ŷ1[y1.==1])
		cost2 = 1-mean(ŷ2[y2.==1])
		trivial_cost1 = 1-mean(y1_triv[y1.==1])
		trivial_cost2 = 1-mean(y2_triv[y2.==1])
	else
		norm_const = fp_rate+fn_rate
		cost1 = fp_rate/norm_cost * mean(ŷ1[y1.==0]) + fn_rate/norm_cost * (1-mean(ŷ1[y1.==1]))
		cost2 = fp_rate/norm_cost * mean(ŷ2[y2.==0]) + fn_rate/norm_cost * (1-mean(ŷ2[y2.==1]))
		trivial_cost1 = fp_rate/norm_cost * mean(y1_triv[y1.==0]) + fn_rate/norm_cost * (1-mean(y1_triv[y1.==1]))
		trivial_cost2 = fp_rate/norm_cost * mean(y2_triv[y2.==0]) + fn_rate/norm_cost * (1-mean(y2_triv[y2.==1]))
	end
	
        if cost2>cost1 && cost2<trivial_cost1
	    	mix_rate1 = (cost2-cost1)/(trivial_cost1-cost1)
		mix_rate2 = 0
	elseif cost1>cost2 && cost1<trivial_cost2
		mix_rate1 = 0
		mix_rate2 = (cost1-cost2)/(trivial_cost2-cost2)
	else 
		mix_rate1 = 0
		mix_rate2 = 0
	end	
	return mix_rate1, mix_rate2
end

