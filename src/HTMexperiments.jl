module HTMexperiments

using HierarchicalTemporalMemory
using Random, Statistics
import Lazy: @>
using UnPack

include("metrics.jl")
export sparseness, entropy, noiserobustness, stabilitySP
include("makedata.jl")
export randomSDR

end
