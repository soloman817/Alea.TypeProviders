module Test.GPUTypes.Test

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open Test.GPUTypes.Data

type BaseHelper = Helper.Test.GPUTypes.Data.Base
type DerivedHelper = Helper.Test.GPUTypes.Data.Derived

let assertArrayEqual (eps:float option) (A:'T[]) (B:'T[]) =
    (A, B) ||> Array.iter2 (fun a b -> eps |> function
        | None -> Assert.AreEqual(a, b)
        | Some eps -> Assert.That(b, Is.EqualTo(a).Within(eps)))

[<Test>]
let ``Pair blob`` () =
    let template = cuda {
        let! kernel =
            <@ fun (output:deviceptr<float>) (input:BaseHelper.FloatPairHelper.TransposedSeq) ->
                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x
                let mutable i = start
                while i < input.Length do
                    output.[i] <- input.First.[i] + input.Second.[i]
                    i <- i + stride @>
            |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            let run (logger:ITimingLogger) (n:int) =
                let hInput = Array.init n (fun i ->
                    { Base.FloatPair.First = TestUtil.genRandomDouble -100.0 100.0 i
                      Base.FloatPair.Second = TestUtil.genRandomDouble -50.0 50.0 i })
                let hOutput = hInput |> Array.map (fun pair -> pair.First + pair.Second)

                use blob = new Blob(worker, logger)
                let dInput = BaseHelper.FloatPairHelper.BlobTransposedSeq.Create(blob, hInput)
                let dOutput = blob.CreateArray<float>(dInput.Length)
                let lp = LaunchParam(16, 512)
                kernel.Launch lp dOutput.Ptr dInput.Device
                let dOutput = dOutput.Gather()

                assertArrayEqual None hOutput dOutput

            run ) }

    use program = template |> Compiler.load Worker.Default
    let logger = TimingLogger("Blob")
    program.Run logger (1<<<24)
    //program.Run logger (1<<<24)
    logger.DumpLogs()

[<Test>]
let ``Generic Pair blob`` () =
    let template = cuda {
        let! kernel =
            <@ fun (output:deviceptr<float>) (input:BaseHelper.PairByInt32AndDoubleHelper.TransposedSeq) ->
                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x
                let mutable i = start
                while i < input.Length do
                    output.[i] <- float(input.First.[i]) + input.Second.[i]
                    i <- i + stride @>
            |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            let run (logger:ITimingLogger) (n:int) =
                let hInput = Array.init n (fun i ->
                    let pair : Base.Pair<int, float> =
                        { First = TestUtil.genRandomSInt32 -100 100 i
                          Second = TestUtil.genRandomDouble -50.0 50.0 i }
                    pair)
                let hOutput = hInput |> Array.map (fun pair -> float(pair.First) + pair.Second)

                use blob = new Blob(worker, logger)
                let dInput = BaseHelper.PairByInt32AndDoubleHelper.BlobTransposedSeq.Create(blob, hInput)
                let dOutput = blob.CreateArray<float>(dInput.Length)
                let lp = LaunchParam(16, 512)
                kernel.Launch lp dOutput.Ptr dInput.Device
                let dOutput = dOutput.Gather()

                assertArrayEqual None hOutput dOutput

            run ) }

    use program = template |> Compiler.load Worker.Default
    let logger = TimingLogger("Blob")
    program.Run logger (1<<<24)
    //program.Run logger (1<<<24)
    logger.DumpLogs()

[<Test>]
let ``Combine blob`` () =
    let template = cuda {
        let! kernel =
            <@ fun (output:deviceptr<float>) (input:DerivedHelper.FloatPairAndTripleHelper.TransposedSeq) ->
                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x
                let mutable i = start
                while i < input.Length do
                    output.[i] <- input.Pair.First.[i] + input.Pair.Second.[i] + input.Triple.First.[i] + input.Triple.Second.[i] + input.Triple.Third.[i]
                    i <- i + stride @>
            |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply kernel

            let run (logger:ITimingLogger) (n:int) =
                let hInput = Array.init n (fun i ->
                    let pair : Base.FloatPair = 
                        { First = TestUtil.genRandomDouble -100.0 100.0 i
                          Second = TestUtil.genRandomDouble -50.0 50.0 i }
                    let triple : Base.FloatTriple =
                        { First = TestUtil.genRandomDouble -100.0 100.0 i
                          Second = TestUtil.genRandomDouble -50.0 50.0 i
                          Third = TestUtil.genRandomDouble -10.0 10.0 i }
                    { Derived.FloatPairAndTriple.Pair = pair
                      Derived.FloatPairAndTriple.Triple = triple } )
                let hOutput = hInput |> Array.map (fun combine -> combine.Pair.First + combine.Pair.Second + combine.Triple.First + combine.Triple.Second + combine.Triple.Third)

                use blob = new Blob(worker, logger)
                let dInput = DerivedHelper.FloatPairAndTripleHelper.BlobTransposedSeq.Create(blob, hInput)
                let dOutput = blob.CreateArray<float>(dInput.Length)
                let lp = LaunchParam(16, 512)
                kernel.Launch lp dOutput.Ptr dInput.Device
                let dOutput = dOutput.Gather()

                assertArrayEqual None hOutput dOutput

            run ) }

    use program = template |> Compiler.load Worker.Default
    let logger = TimingLogger("Blob")
    program.Run logger (1<<<20)
    //program.Run logger (1<<<24)
    logger.DumpLogs()

[<Test>]
let ``Array field blob``() =
    let template = cuda {
        let! kernel =
            <@ fun (output1:deviceptr<float>) (output2:deviceptr<int>) (input:BaseHelper.FloatNestedArrayHelper.TransposedSeq) ->
                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x
                let mutable i = start
                let n = input.Length
                while i < n do
                    let mutable sum = input.Offset1.[i]
                    for j = 0 to input.Array1.Length(i) - 1 do
                        sum <- sum + input.Array1.[i, j]
                    output1.[i] <- sum

                    let mutable sum = input.Offset2.[i]
                    for j = 0 to input.Array2.Length(i) - 1 do
                        sum <- sum + input.Array2.[i, j]
                    output2.[i] <- sum

                    i <- i + stride @>
            |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply kernel
            let lp = LaunchParam(32, 256)

            let run (logger:ITimingLogger) =
                use blob = new Blob(worker, logger)

                let input' = Base.FloatNestedArray.RandomArray(1000)
                let output1' = input' |> Array.map (fun x -> x.Offset1 + (x.Array1 |> Array.sum))
                let output2' = input' |> Array.map (fun x -> x.Offset2 + (x.Array2 |> Array.sum))

                let input = BaseHelper.FloatNestedArrayHelper.BlobTransposedSeq.Create(blob, input')
                let output1 = blob.CreateArray<float>(input.Length)
                let output2 = blob.CreateArray<int>(input.Length)

                kernel.Launch lp output1.Ptr output2.Ptr input.Device
                assertArrayEqual (Some 1e-7) output1' (output1.Gather())
                assertArrayEqual None output2' (output2.Gather())
                TestUtil.testLaunchingTime worker "3" 5000 input.Device

            run ) }

    let worker = Worker.Default
    use program = template |> Compiler.load worker
    let logger = TimingLogger("Blob")
    worker.Eval <| fun _ -> program.Run logger
    logger.DumpLogs()

[<Test>]
let ``Generic Array field blob``() =
    let template = cuda {
        let! kernel =
            <@ fun (output1:deviceptr<float>) (output2:deviceptr<int>) (input:BaseHelper.NestedArrayByDoubleAndInt32Helper.TransposedSeq) ->
                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x
                let mutable i = start
                let n = input.Length
                while i < n do
                    let mutable sum = input.Offset1.[i]
                    for j = 0 to input.Array1.Length(i) - 1 do
                        sum <- sum + input.Array1.[i, j]
                    output1.[i] <- sum

                    let mutable sum = input.Offset2.[i]
                    for j = 0 to input.Array2.Length(i) - 1 do
                        sum <- sum + input.Array2.[i, j]
                    output2.[i] <- sum

                    i <- i + stride @>
            |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply kernel
            let lp = LaunchParam(32, 256)

            let run (logger:ITimingLogger) =
                use blob = new Blob(worker, logger)

                let input' = Base.NestedArray<float, int>.RandomArray(1000)
                let output1' = input' |> Array.map (fun x -> x.Offset1 + (x.Array1 |> Array.sum))
                let output2' = input' |> Array.map (fun x -> x.Offset2 + (x.Array2 |> Array.sum))

                let input = BaseHelper.NestedArrayByDoubleAndInt32Helper.BlobTransposedSeq.Create(blob, input')
                let output1 = blob.CreateArray<float>(input.Length)
                let output2 = blob.CreateArray<int>(input.Length)

                kernel.Launch lp output1.Ptr output2.Ptr input.Device
                assertArrayEqual (Some 1e-7) output1' (output1.Gather())
                assertArrayEqual None output2' (output2.Gather())
                TestUtil.testLaunchingTime worker "3" 5000 input.Device

            run ) }

    let worker = Worker.Default
    use program = template |> Compiler.load worker
    let logger = TimingLogger("Blob")
    worker.Eval <| fun _ -> program.Run logger
    logger.DumpLogs()

[<Test>]
let ``Equal length array field blob``() =
    let template = cuda {
        let! kernel =
            <@ fun (output1:deviceptr<float>) (output2:deviceptr<int>) (input:BaseHelper.FloatNestedEqualLengthArrayHelper.TransposedSeq) ->
                __debug(input.Array1)
                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x
                let mutable i = start
                let n = input.Length
                while i < n do
                    let mutable sum = input.Offset1.[i]
                    for j = 0 to input.Array1.Length(i) - 1 do
                        sum <- sum + input.Array1.[i, j]
                    output1.[i] <- sum

                    let mutable sum = input.Offset2.[i]
                    for j = 0 to input.Array2.Length(i) - 1 do
                        sum <- sum + input.Array2.[i, j]
                    output2.[i] <- sum

                    i <- i + stride @>
            |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply kernel
            let lp = LaunchParam(32, 256)

            let run (logger:ITimingLogger) =
                use blob = new Blob(worker, logger)

                let input' = Base.FloatNestedEqualLengthArray.RandomArray(1000)
                let output1' = input' |> Array.map (fun x -> x.Offset1 + (x.Array1 |> Array.sum))
                let output2' = input' |> Array.map (fun x -> x.Offset2 + (x.Array2 |> Array.sum))

                let input = BaseHelper.FloatNestedEqualLengthArrayHelper.BlobTransposedSeq.Create(blob, input')
                let output1 = blob.CreateArray<float>(input.Length)
                let output2 = blob.CreateArray<int>(input.Length)

                kernel.Launch lp output1.Ptr output2.Ptr input.Device
                assertArrayEqual (Some 1e-7) output1' (output1.Gather())
                assertArrayEqual None output2' (output2.Gather())
                TestUtil.testLaunchingTime worker "3" 5000 input.Device

            run ) }

    let worker = Worker.Default
    use program = template |> Compiler.load worker
    let logger = TimingLogger("Blob")
    worker.Eval <| fun _ -> program.Run logger
    logger.DumpLogs()

