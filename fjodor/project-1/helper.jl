module Helper
using DataFrames
using Plots
using MLJ
using CategoricalArrays
using CategoricalDistributions

export Meas, get_meas_as_tuple, get_measurements, plot_roccurve, get_machine, fit_predict

mutable struct Meas
    model::String
    brier_loss::Float64
    auc::Float64
    accuracy::Float64
end

function get_meas_as_tuple(meas)
    return (meas.model, meas.brier_loss, meas.auc, meas.accuracy)
end

function get_measurements(model_name::String, 
                          ŷ::UnivariateFiniteVector{OrderedFactor{2}, Int64, UInt32, Float64},
                          y::CategoricalVector{Int64, UInt32, Int64, CategoricalValue{Int64, UInt32}, Union{}},
                          mt::Vector{String})::Meas
    meas = Meas(model_name, 0.0, 0.0, 0.0)
    if "brier_loss" ∈ mt
    meas.brier_loss = round(brier_loss(ŷ, y) |> mean, digits=3)
    end
    if "auc" ∈ mt
    meas.auc = round(auc(ŷ, y), digits=3)
    end
    if "accuracy" ∈ mt
    meas.accuracy = round(accuracy(mode.(ŷ), y), digits=3)
    end
    return meas
end

function plot_roccurve(ŷ::UnivariateFiniteVector{OrderedFactor{2}, Int64, UInt32, Float64}, y::CategoricalVector{Int64, UInt32, Int64, CategoricalValue{Int64, UInt32}, Union{}})
    roc_curve = roc(ŷ, y)
    plt = scatter(roc_curve, legend=false)
    plot!(plt, xlab="false positive rate", ylab="true positive rate")
    plot!([0, 1], [0, 1], linewidth=2, linestyle=:dash, color=:black)
end

function get_machine(model, X::DataFrame, y::CategoricalVector{Int64, UInt32, Int64, CategoricalValue{Int64, UInt32}, Union{}})::Machine
    _model = model
    show(stdout, "text/plain", _model)
    return machine(_model, X, y)
 end

function fit_predict(machine, train::Vector{Int64}, validation::Vector{Int64})::UnivariateFiniteVector{OrderedFactor{2}, Int64, UInt32, Float64}
    fit!(machine, rows=train)
    ŷ = predict(machine, rows=validation)
    return ŷ
end

end