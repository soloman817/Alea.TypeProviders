namespace Test.GPUTypes.Data.Base

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.TypeProviders.GPUHelperDirectives

[<Record;GenGPUHelper>]
type FloatPair =
    {
        First : float
        Second : float
    }

//[<Record;GenGPUHelper>]
type FloatTriple =
    {
        First : float
        Second : float
        Third : float
    }

[<Record>]
[<GenGPUHelperBy([| typeof<int>; typeof<float> |])>]
[<GenGPUHelperBy([| typeof<float>; typeof<float32> |])>]
[<GenGPUHelperBy([| typeof<float>; typeof<float> |])>]
type Pair<'T1, 'T2> =
    {
        First : 'T1
        Second : 'T2
    }


