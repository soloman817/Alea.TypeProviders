module Test.GPUTypes.Test

open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework

type GPUPair = Test.GPUTypes.Data.Base.FloatPair
type GPUPairSeq = Test.GPUTypes.Data.Helper.Test.GPUTypes.Data.Base.FloatPairHelper.TransposedSeq

type GPUTriple = Test.GPUTypes.Data.Base.FloatTriple
type GPUTripleSeq = Test.GPUTypes.Data.Helper.Test.GPUTypes.Data.Base.FloatTripleHelper.TransposedSeq

type GPUCombine = Test.GPUTypes.Data.Derived.FloatPairAndTriple
type GPUCombineSeq = Test.GPUTypes.Data.Helper.Test.GPUTypes.Data.Derived.FloatPairAndTripleHelper.TransposedSeq

let assertArrayEqual (eps:float option) (A:'T[]) (B:'T[]) =
    (A, B) ||> Array.iter2 (fun a b -> eps |> function
        | None -> Assert.AreEqual(a, b)
        | Some eps -> Assert.That(b, Is.EqualTo(a).Within(eps)))

[<Test>]
let ``Pair blob`` () =
    let template = cuda {
        let! kernel =
            <@ fun (output:deviceptr<float>) (input:GPUPairSeq) ->
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
                    { GPUPair.First = TestUtil.genRandomDouble -100.0 100.0 i
                      GPUPair.Second = TestUtil.genRandomDouble -50.0 50.0 i })
                let hOutput = hInput |> Array.map (fun pair -> pair.First + pair.Second)

                use blob = new Blob(worker, logger)
                let dInput = GPUPairSeq.BlobCreate(blob, hInput)
                let dOutput = blob.CreateArray<float>(dInput.Length)
                let lp = LaunchParam(16, 512)
                kernel.Launch lp dOutput.Ptr (GPUPairSeq.BlobTrigger(dInput))
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
            <@ fun (output:deviceptr<float>) (input:GPUCombineSeq) ->
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
                    let pair = 
                        { GPUPair.First = TestUtil.genRandomDouble -100.0 100.0 i
                          GPUPair.Second = TestUtil.genRandomDouble -50.0 50.0 i }
                    let triple =
                        { GPUTriple.First = TestUtil.genRandomDouble -100.0 100.0 i
                          GPUTriple.Second = TestUtil.genRandomDouble -50.0 50.0 i
                          GPUTriple.Third = TestUtil.genRandomDouble -10.0 10.0 i }
                    { GPUCombine.Pair = pair
                      GPUCombine.Triple = triple } )
                let hOutput = hInput |> Array.map (fun combine -> combine.Pair.First + combine.Pair.Second + combine.Triple.First + combine.Triple.Second + combine.Triple.Third)

                use blob = new Blob(worker, logger)
                let dInput = GPUCombineSeq.BlobCreate(blob, hInput)
                let dOutput = blob.CreateArray<float>(dInput.Length)
                let lp = LaunchParam(16, 512)
                kernel.Launch lp dOutput.Ptr (GPUCombineSeq.BlobTrigger(dInput))
                let dOutput = dOutput.Gather()

                assertArrayEqual None hOutput dOutput

            run ) }

    use program = template |> Compiler.load Worker.Default
    let logger = TimingLogger("Blob")
    program.Run logger (1<<<20)
    //program.Run logger (1<<<24)
    logger.DumpLogs()
