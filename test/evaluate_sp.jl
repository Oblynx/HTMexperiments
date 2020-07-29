using CSV, HierarchicalTemporalMemory, Plots, Setfield, Lazy, HypothesisTests, Statistics
    using HTMexperiments #debug
import Random: seed!
seed!(0)
theme(:default, titlefontsize=10, xguidefontsize=9, yguidefontsize=9, xtickfontsize=7, ytickfontsize=7,
    guidefontfamily = "Helvetica", tickfontfamily = "Helvetica")
gr(size = (1300, 731))

# Create a few SpatialPoolers
szᵢₙ= (32,32)
spTopo= SpatialPooler(SPParams(
  szᵢₙ= szᵢₙ,
  szₛₚ= (32,32),
  γ= 5,
  s= 0.02,
  prob_synapse=1,
  β= 100,
  Tboost= 1000,
))
sp1d= SpatialPooler(SPParams(
  szᵢₙ= prod(szᵢₙ),
  szₛₚ= (1024,),
  s= 0.025,
  prob_synapse=1,
  β= 100,
  Tboost= 1000,
  enable_local_inhibit= false,
))
sp2= SpatialPooler(SPParams(
  szᵢₙ=1024,
  szₛₚ=1024,
  γ=1024,
  s=0.025,
  prob_synapse=0.25,
  θ_stimulus_activate=3,
  p⁺_01= 0.15,
  p⁻_01= 0.07,
  β=10,
  Tboost= 200,
  enable_local_inhibit= false,
))

# random sparse inputs
n= 255  # samples
spToEval= sp1d
train= Dict(); g= Dict(); m= Dict()
train[:epochs]= 50
train[:samples]= [0,2,5,18,40,50]
beforeAfter= [0,train[:samples][end]]

m["entropy"]= zeros(train[:samples]|>length,15)
m["noisemetric"]= zeros(train[:samples]|>length,10)
noiseCurves= zeros(20,train[:samples]|>length)
noiseRange= zeros(20)

data= randomSDR(n)
a= Array{Bool}(undef, prod(spToEval.params.szₛₚ),n)
(i-> @views a[:,i].= spToEval(data[:,i])|> vec).(1:n)
sparse_z= (i-> data[:,i]|> sparseness).(1:n)
sparse_a= (i-> a[:,i]|> sparseness).(1:n)

# Sparsity control
g["data"]= plot(heatmap(data[1:200,:]', ylabel="inputs", title="input vectors", colorbar=false, c=:bone),
                heatmap(a[1:200,:]', ylabel="inputs", xlabel="t", title="SP outputs", colorbar=false, c=:bone), layout= (2,1))
g["sparsity"]= plot(plot(sparse_z, 1:n, xlims=(0,0.20), xticks= [0,0.1,0.20]),
                    plot(sparse_a, 1:n, xlims=(0,0.20), xticks= [0.02,0.1,0.20]), layout= (2,1), title="sparsity", legend=nothing)

# Entropy: generate new data and train the SP for 10 independent experiments for entropy
let sp= deepcopy(spToEval), t=1
  m["entropy"][1,:]= entropy(sp, data, trials=15)
  # train the SP over a few epochs
  for e= 1:train[:epochs]
    (i-> step!(sp, @> data[:,i] reshape(szᵢₙ))).(1:n)
    e ∈ train[:samples] && begin
      t= t+1
      m["entropy"][t,:]= entropy(sp, data, trials=15)
    end
  end
end
entropyTest= OneSampleTTest(m["entropy"][1,:], m["entropy"][end,:])|> display
m["entropy"]= Dict("mean" => [m["entropy"][1,:], m["entropy"][end,:]].|>mean,
                   "max"  => mapreduce(i->entropy(randomSDR(n,sparsity=0.02)),+,1:10)/10)
g["entropy"]= bar(beforeAfter, m["entropy"]["mean"], ylabel="entropy (bits)", xlabel="epoch", xticks=beforeAfter, legend=nothing)
hline!([m["entropy"]["max"]], linestyle=:dash, ylims=(0,m["entropy"]["max"]*1.02))

# Noise robustness
noiseredNoiseEst= zeros(train[:samples]|> length)
let sp= deepcopy(spToEval), t=1
  m["noisemetric"][1,:], noiseCurves[:,1], noiseredNoiseEst[1], noiseRange[:]= noiserobustness(sp, data)
  # train the SP over a few epochs
  for e= 1:train[:epochs]
    (i-> step!(sp, @> data[:,i] reshape(szᵢₙ))).(1:n)
    e ∈ train[:samples] && begin
      t= t+1
      m["noisemetric"][t,:], noiseCurves[:,t], noiseredNoiseEst[t]= noiserobustness(sp, data)
    end
  end
end
noiseTest= OneSampleTTest(m["noisemetric"][1,:], m["noisemetric"][end,:])|> display
m["noisemetric"]= [m["noisemetric"][1,:], m["noisemetric"][end,:]].|>mean
g["noisemetric"]= bar(beforeAfter, m["noisemetric"], ylabel="noise robustness", xlabel="epoch", xticks=beforeAfter, legend=nothing)
g["noisecurves"]= plot(noiseRange, noiseCurves[:,1], labels="epoch ".*(train[:samples][1]|>string), ribbon=noiseredNoiseEst[1],
                       ylabel="SP stability", xlabel="noise level")
foreach(2:length(train[:samples])-1) do i
  plot!(noiseRange, noiseCurves[:,i], labels="epoch ".*(train[:samples][i]|>string), ribbon=noiseredNoiseEst[i] )
end
# labels=reshape("epoch ".*([0,2,5,18,40].|>string),1,:)
plot!(noiseRange,noiseRange|>reverse, linestyle=:dash, label=nothing)

## activation frequency
aFreq= zeros(size(a,1),2)
let sp= deepcopy(spToEval)
  aFreq[:,1]= sum(a, dims=2)
  for e= 1:train[:epochs]
    (i-> step!(sp, @> data[:,i] reshape(szᵢₙ))).(1:n)
  end
  (i-> @views a[:,i].= sp(data[:,i])|> vec).(1:n)
  aFreq[:,2]= sum(a, dims=2)
end
g["activation freq"]= plot(histogram(aFreq[:,1], title="epoch 0"),
                           histogram(aFreq[:,2], title="epoch $(train[:samples][end])"), xlabel="activation frequency", ylabel="fraction of SP columns", legend=nothing)

## Compose final plot
m|> display
layout= @layout [a{.30w} b{.1w} grid(2,1){.1w} [e; f]]
g["SP metrics"]= plot(g["data"],g["sparsity"],g["entropy"],g["noisemetric"], g["activation freq"], g["noisecurves"], layout=layout)
savefig(g["SP metrics"], "sp_metrics.svg")