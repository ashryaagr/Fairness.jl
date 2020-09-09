using Fairness, MLJ, PrettyPrinting
using DataFrames

# Score on training data to assess model quality
function get_scores(m, X, y, rows, set)
  yhat = predict(m, rows=rows)
  ft = fair_tensor(yhat, y[rows], X[rows, :z])
  df = disparity(
    [accuracy, false_positive_rate],
    ft, refGrp="A",
    func=(x, y)->(x-y)
  )
  df[:,"accuracy"] .= accuracy(ft)
  df[:,"false_positive_rate"] .= false_positive_rate(ft)
  df[:,"set"] .= set
  return df
end

function run_synthetic_experiment(m, ntrain = 10000, genFun = genZafarData; genargs...)
  (k, m) = m

  ntest = 1000
  repls = 30

  # Train data
  n = ntrain + (ntest * repls)
  X,y = genFun(n; genargs...)

  # Fit classifier
  #@load NeuralNetworkClassifier
  #model = @pipeline ContinuousEncoder NeuralNetworkClassifier
  train, test = partition(eachindex(y), ntrain/n, shuffle=true)
  m = machine(m2, X, y)
  fit!(m, rows=train)

  # Compute scores on train data
  df = get_scores(m, X, y, train, "insample")

  # Score on different test subsets
  tsts = [test[i:(i+ntest-1)] for i in 1:ntest:length(test)]
  for tst in tsts
    append!(df, get_scores(m, X, y, tst, "oob"*string(minimum(tst))))
  end
  df = df[df.labels .== "B",:]
  select!(df, Not(1))
  df[:, "n"] .= ntrain

  return df
end


@load RandomForestClassifier pkg=DecisionTree
model = @pipeline ContinuousEncoder RandomForestClassifier
m1 = ReweighingSamplingWrapper(classifier=model, grp=:z)
m2 = LinProgWrapper(classifier=m1, grp=:z, measure=false_positive_rate)
m3 = EqOddsWrapper(classifier=m1, grp=:z)
clf = Dict("RF" => model, "RF_RW" => m1, "RF_RW_RP" => m2, "RF_RW_EO" => m3)

res = DataFrame()
for c in clf
    append!(res, run_synthetic_experiment(c, 1000))
end

using StatsPlots
@df res scatter(:accuracy, :false_positive_rate_disparity)
