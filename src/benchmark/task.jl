mutable struct Task # It has all what we need to make predictions
	X
	y
	grp
	debiasmeasures

	function Task(X, y, grp, args...)
		cutoff = 0.05
		# groups with representation less than cutoff won't be appearing
		mask = ones(Bool, length(y))
		for group in levels(X[!, grp])
			if sum(X[!, grp].==group) < cutoff*length(y)
				mask[X[!, grp].==group] .= 0
			end
		end
		X, y = X[mask, :], y[mask]
		X[!, grp] = categorical(convert(Array, X[!, grp]))
		return new(X, y, grp, args...)
	end
end

function Task(X, y; grp, debiasmeasures)
	return Task(X, y, grp, debiasmeasures)
end

mutable struct FairnessProblem #It has all what we need to measure fairness using task
	task::Task
	refGrp # Reference Group for disparity calculation
	measures #Measures to evaluate apart from accuracy
	repls # replications
	nfolds # Number of folds for cross validation
	seed # Random Seed
	name # Name of the file or a unique name to indicate the experiment
end

function FairnessProblem(task::Task; refGrp=nothing,
								   measures=nothing, repls=10, nfolds=10,
								   seed=5, name=nothing)
   name = name==nothing ? randstring(['A':'Z'; '0':'9'], 12) : name
   measures = measures==nothing ? debiasmeasures : measures
   refGrp = refGrp==nothing ? StatsBase.mode(task.X[!, task.grp]) : refGrp
   task.X[!, task.grp] = convert(Array{Any}, task.X[!, task.grp])
   task.X[!, task.grp] = string.(task.X[!, task.grp])
   task.X[task.X[!, task.grp].!=refGrp, task.grp] .= "0"
   task.X[task.X[!, task.grp].==refGrp, task.grp] .= "1"
   task.X[!, task.grp] = categorical(convert(Array{String},
												task.X[!, task.grp]))
   transform!(task.X, task.grp => categorical, renamecols=false)
   return FairnessProblem(task, refGrp, measures, repls, nfolds, seed, name)
end

function FairnessProblem(X, y; grp, debiasmeasures, refGrp=nothing,
							measures=nothing, repls=10, nfolds=10,
							seed=5, name=nothing)
	return FairnessProblem(Task(X, y, grp, debiasmeasures),
							refGrp=refGrp, measures=measures, repls=repls,
							nfolds=nfolds, seed=seed, name=name)
end
