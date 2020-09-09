using Fairness, MLJ, PrettyPrinting

ntrain = 10000
ntest = 1000
repls = 30

# Train data
n = ntrain + (ntest * repls)
X,y = genZafarData(n)

@load NeuralNetworkClassifier
builder = MLJFlux.Short(dropout=0.1)
model = @pipeline ContinuousEncoder NeuralNetworkClassifier(builder = builder)
m1 = ReweighingSamplingWrapper(classifier=model, grp=:z)
m2 = LinProgWrapper(classifier=m1, grp=:z, measure=false_positive_rate)
train, test = partition(eachindex(y), ntrain/n, shuffle=true)
m = machine(m2, X, y)
fit!(m, rows=train)

# Score on training data to assess model quality
yhat = predict(m, rows=train)
ft = fair_tensor(yhat, y[train], X[train, :z])
df_disparity = disparity(
  [accuracy, false_positive_rate],
  ft, refGrp="1",
  func=(x, y)->(x-y)/y
)
df_disparity[df_disparity.labels .== "0",:] |> pprint

accuracy(ft)

# Score on different test subsets
tsts = [test[i:(i+ntest-1)] for i in 1:ntest:length(test)]
df = DataFrame()
for tst in tsts
  yhat = predict(m, rows=tst)
  ft = fair_tensor(yhat, y[tst], X[tst, :z])
  df_disparity = disparity(
    [accuracy, false_positive_rate],
    ft, refGrp="1",
    func=(x, y)->(x-y)/y
  )
  append!(df, df_disparity)
end
df = df[df.labels .== "0",:]
combine(df,
  :accuracy_disparity => mean, :accuracy_disparity => std,
  :false_positive_rate_disparity => mean, :false_positive_rate_disparity => std
) |> pprint
