module HTMexperiments

using HierarchicalTemporalMemory
using Random
import Lazy: @>
using UnPack

include("metrics.jl")
export sparseness, entropySP, noiserobustness, stabilitySP
include("makedata.jl")
export randomSDR

end
