namespace Test.GPUTypes.Data.Base

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.TypeProviders.GPUHelperDirectives

[<Record;TransposedSeq>]
type FloatPair =
    {
        First : float
        Second : float
    }

//[<Record;TransposedSeq>]
type FloatTriple =
    {
        First : float
        Second : float
        Third : float
    }