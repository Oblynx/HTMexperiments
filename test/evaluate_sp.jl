using CSV, HierarchicalTemporalMemory, Plots, Setfield, Lazy, HypothesisTests
import Random: seed!
seed!(0)

# Create a few SpatialPoolers
szᵢₙ= (32,32)
spTopo= SpatialPooler(SPParams(
  szᵢₙ= szᵢₙ,
  szₛₚ= (32,32),
  γ= 5,
  s= 0.02,
  β= 100,
  Tboost= 1000,
))
sp1d= SpatialPooler(SPParams(
  szᵢₙ= prod(szᵢₙ),
  szₛₚ= (1024,),
  s= 0.02,
  β= 100,
  Tboost= 1000,
  enable_local_inhibit= false,
))
sp2= SpatialPooler(SPParams(
  szᵢₙ=1024,
  szₛₚ=1024,
  γ=1024,
  s=0.02,
  prob_synapse=0.25,
  θ_stimulus_activate=3,
  p⁺_01= 0.15,
  p⁻_01= 0.07,
  β=10,
  Tboost= 200,
  enable_local_inhibit= false,
))

    using HTMexperiments #debug
## random sparse inputs
using Statistics
n= 200  # samples
data= randomSDR(n)
a= Array{Bool}(undef, prod(sp1d.params.szₛₚ),n)
(i-> @views a[:,i].= sp1d(data[:,i])|> vec).(1:n)
sparse_z= (i-> data[:,i]|> sparseness).(1:n)
sparse_a= (i-> a[:,i]|> sparseness).(1:n)

g= Dict(); m= Dict()
m["entropy"]= zeros(2,15)
m["noisemetric"]= zeros(2,10)
noiseCurves= zeros(41,5)

m["entropy"][1,1]= entropySP(a, tdim=2)
g["data"]= plot(heatmap(data[1:200,:]', ylabel="inputs", title="Input vectors", colorbar=false, c=:bone),
                heatmap(a[1:200,:]', xlabel="t", title="SP outputs", colorbar=false, c=:bone), layout= (2,1))
g["sparsity"]= plot(plot(sparse_z, 1:n, xlims=(0,0.2), xticks= [0,0.1,0.2]),
                     plot(sparse_a, 1:n, xlims=(0,0.2), xticks= [0,0.1,0.2]), layout= (2,1))

## generate new data and train the SP for 10 independent experiments for entropy
let sp= deepcopy(sp1d)
  for i= 1:15
    data= randomSDR(n)
    (i-> @views a[:,i].= sp(data[:,i])|> vec).(1:n)
    m["entropy"][1,i]= entropySP(a, tdim=2)
    (i-> step!(sp, @> data[:,i] reshape(szᵢₙ))).(1:n)
    (i-> @views a[:,i].= sp(data[:,i])|> vec).(1:n)
    m["entropy"][2,i]= entropySP(a, tdim=2)
  end
end
entropyTest= OneSampleTTest(m["entropy"][1,:], m["entropy"][2,:])|> display
m["entropy"]= Dict("mean" => [m["entropy"][1,:], m["entropy"][2,:]].|>mean,
                   "max"  => mapreduce(i->entropySP(randomSDR(n,sparsity=0.02),tdim=2),+,1:10)/10)

g["entropy"]= bar(m["entropy"]["mean"])
hline!([m["entropy"]["max"]], linestyle=:dash)

## noise curves
let sp= deepcopy(sp1d), k=1
  m["noisemetric"][1,:], noiseCurves[:,1]= noiserobustness(sp, data)
  # train the SP over a few epochs
  for e= 1:40
    (i-> step!(sp, @> data[:,i] reshape(szᵢₙ))).(1:n)
    e ∈ [0,2,5,18,40] && begin
      k= k+1
      m["noisemetric"][2,:], noiseCurves[:,k]= noiserobustness(sp, data)
    end
  end
end
noiseTest= OneSampleTTest(m["noisemetric"][1,:], m["noisemetric"][2,:])|> display
#m["noisemetric"]= [m["noisemetric"][1,:], m["noisemetric"][2,:]].|>mean
#g["noisemetric"]= bar(m["noisemetric"])
g["noisecurves"]= plot(0:1/40:1, noiseCurves, labels=reshape("epoch ".*([0,2,5,18,40].|>string),1,:) )
##


m|> display
plot(g["data"],g["sparsity"],g["entropy"],g["noisemetric"] g["noisecurves"], layout= @layout [a{.5w} b{.2w} grid(2,1){.2w} d{.5w}])