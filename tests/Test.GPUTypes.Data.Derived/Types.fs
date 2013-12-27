namespace Test.GPUTypes.Data.Derived

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.TypeProviders.GPUHelperDirectives
open Test.GPUTypes.Data.Base

[<Record;GenGPUHelper>]
type FloatPairAndTriple =
    {
        Pair : FloatPair
        Triple : FloatTriple
    }