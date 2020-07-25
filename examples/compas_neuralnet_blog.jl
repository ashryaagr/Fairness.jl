using Fairness
using MLJ
using PrettyPrinting

data = @load_compas
X, y = data

@load NeuralNetworkClassifier
model = @pipeline ContinuousEncoder NeuralNetworkClassifier

wrappedModel = ReweighingSamplingWrapper(classifier=model, grp=:race)
wrappedModel2 = LinProgWrapper(classifier=wrappedModel, grp=:race,
                                measure=false_positive_rate)

result = evaluate(wrappedModel2,
  X, y,
  measures=MetricWrappers(
    [true_positive_rate, false_positive], grp=:race)) |> pprint


train, test = partition(eachindex(y), 0.7, shuffle=true)
mach = machine(wrappedModel2, X, y)
fit!(mach, rows=train)


ŷ = predict(mach, rows=test)
ŷ |> pprint


ft = fair_tensor(ŷ, y[test], X[test, :gender_status])

df_disparity = disparity(
  [true_positive_rate, positive_predictive_value],
  ft, refGrp="male_single",
  func=(x, y)->(x-y)/y
)

parity(df_disparity,
  func= (x) -> abs(x)<0.07
)
