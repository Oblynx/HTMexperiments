using CSV, HierarchicalTemporalMemory, Plots

spTopo= SpatialPooler(SPParams(
  szᵢₙ= (32,32),
  szₛₚ= (32,32),
  γ= 5,
  s= 0.02,
  β= 100,
  Tboost= 1000,
))
sp1d= SpatialPooler(SPParams(
  szᵢₙ= (32,32),
  szₛₚ= (1024,),
  γ= 5,
  s= 0.02,
  β= 100,
  Tboost= 1000,
))

## random sparse inputs
data= randomSDR(100)
heatmap(data[1:200,:])