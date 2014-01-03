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

[<GenGPUHelper>]
type FloatNestedArray =
    {
        Offset1 : float
        Offset2 : int
        Array1 : float[]
        Array2 : int[]
    }

    static member RandomArray(length:int) =
        Array.init<FloatNestedArray> length (fun i ->
            let n1 = TestUtil.genRandomSInt32 20 100 ()
            let n2 = TestUtil.genRandomSInt32 20 100 ()
            {
                Offset1 = TestUtil.genRandomDouble -10.0 10.0 i
                Offset2 = TestUtil.genRandomSInt32 -10 10 i
                Array1 = Array.init n1 (TestUtil.genRandomDouble -10.0 10.0)
                Array2 = Array.init n2 (TestUtil.genRandomSInt32 -10 10)
            })

[<GenGPUHelperBy([| typeof<float>; typeof<int> |])>]
type NestedArray<'T1, 'T2> =
    {
        Offset1 : 'T1
        Offset2 : 'T2
        Array1 : 'T1[]
        Array2 : 'T2[]
    }

    static member RandomArray(length:int) =
        Array.init<NestedArray<float, int>> length (fun i ->
            let n1 = TestUtil.genRandomSInt32 20 100 ()
            let n2 = TestUtil.genRandomSInt32 20 100 ()
            {
                Offset1 = TestUtil.genRandomDouble -10.0 10.0 i
                Offset2 = TestUtil.genRandomSInt32 -10 10 i
                Array1 = Array.init n1 (TestUtil.genRandomDouble -10.0 10.0)
                Array2 = Array.init n2 (TestUtil.genRandomSInt32 -10 10)
            })

[<GenGPUHelper>]
type FloatNestedEqualLengthArray =
    {
        Offset1 : float
        Offset2 : int
        [<EqualLengthArray>] Array1 : float[]
        [<EqualLengthArray>] Array2 : int[]
    }

    static member RandomArray(length:int) =
        let n1 = TestUtil.genRandomSInt32 20 100 ()
        let n2 = TestUtil.genRandomSInt32 20 100 ()
        Array.init<FloatNestedEqualLengthArray> length (fun i ->
            {
                Offset1 = TestUtil.genRandomDouble -10.0 10.0 i
                Offset2 = TestUtil.genRandomSInt32 -10 10 i
                Array1 = Array.init n1 (TestUtil.genRandomDouble -10.0 10.0)
                Array2 = Array.init n2 (TestUtil.genRandomSInt32 -10 10)
            })

