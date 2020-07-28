using Fairness
using MLJBase, MLJModels

data = @load_compas
X, y = data

@load RandomForestClassifier pkg=DecisionTree
model = @pipeline ContinuousEncoder RandomForestClassifier

wrappedModel = ReweighingSamplingWrapper(classifier=model, grp=:race)
wrappedModel2 = LinProgWrapper(classifier=wrappedModel, grp=:race,
                                measure=false_positive_rate)

result = evaluate(wrappedModel2,
    X, y,
    measures=[
    Disparity(false_positive_rate, refGrp="Caucasian", grp=:race),
    MetricWrapper(accuracy, grp=:race)])

result_1 = evaluate(model,
    X, y,
    measures=[
    Disparity(false_positive_rate, refGrp="Caucasian", grp=:race),
    MetricWrapper(accuracy, grp=:race)])

using VegaLite
using DataFrames
n_grps = length(levels(X[!, :race]))
dispVals = collect(values(result.measurement[1]))
dispVals_1 = collect(values(result_1.measurement[1]))
df = DataFrame(disparity=vcat(dispVals, dispVals_1),
  algo=vcat(repeat(["Wrapped Model"], n_grps+1), repeat(["ML Model"], n_grps+1)),
  grp=repeat(collect(keys(result.measurement[1])), 2))

df |> @vlplot(
  :bar,
  column="grp:o",
  y={"disparity:q",axis={title="False Positive Rate Disparity"}},
  x={"algo:o", axis={title=""}},
  color={"algo:o"},
  spacing=20,
  config={
  view={stroke=:transparent},
  axis={domainWidth=1}
  }
)
