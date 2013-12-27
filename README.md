Alea.TypeProviders
==================

This project tends to be a collection of type providers. Now it only have one provider:
`GPUHelperProvider`.

GPUHelperProvider
-----------------

### Problem

In coding GPU, sometimes we need take care about the hardware architecture. GPU uses SIMD
instruction, means, there are multiple threads running at same time. This brings a problem
for memory access. Say if we have a struct like this:

```fsharp
type Record = { FieldA : int; FieldB : float }
```

And then check this kernel:
~~~fsharp
let kernel (output:deviceptr<float>) (input:deviceptr<Record>) =
    let tid = threadIdx.x
    output.[tid] <- float(input.[tid].FieldA) + input.[tid].FieldB
~~~

The problem happens when there are multiple threads (a warp 32) running at same time. So the
first memory read `input.[tid].FieldA` will access memory of 0+4, 16+4, 32+4, etc. The struct
size would be 16 bytes (because of alignment, there will be 4 bytes padding after FieldA). Then
the second read will be 8+8, 24+8, 40+8, etc. So there is gap in each memory access operation,
which will make a big problem in GPU, and let hardware caching nearly no use. Hardware tends to
read a continuous memory, as a cache line. Obviously in this case, many reading is wasted.

There is one solution, tranposing data before they go to GPU memory. Image there is a struct:

```fsharp
type RecordTransposedSeq = { FieldAs : deviceptr<int>; FieldBs : deviceptr<float> }
```

Then this time, each read will be continous. And you need also code a host transposing function.
Take the Blob system we use, it would look like:

```fsharp
type Data1 =
    {
        Number1 : float
        Number2 : float
        Integer : int
    }

[<Record>]
type DeviceData1Seq =
    {
        Length : int
        Number1s : deviceptr<float>
        Number2s : deviceptr<float>
        Integers : deviceptr<int>
    }

type Blob with
    member this.CreateData1Seq(host:Data1 seq) =
        let host = host |> Seq.toArray
        let length = host.Length
        let number1s = host |> Array.map (fun x -> x.Number1) |> this.CreateArray
        let number2s = host |> Array.map (fun x -> x.Number2) |> this.CreateArray
        let integers = host |> Array.map (fun x -> x.Integer) |> this.CreateArray
        let bmem : Lazy<DeviceData1Seq> =
            Lazy.Create <| fun _ ->
                {
                    Length = length
                    Number1s = number1s.Ptr
                    Number2s = number2s.Ptr
                    Integers = integers.Ptr
                }
        length, bmem
```

Then here we invovles 3 types:

- The struct type itself : `Data1`
- The transposed seq type used in device : `DeviceData1Seq`. In this provider, it will be called
  `Data1Helper.TransposedSeq`
- A blob memory type which has two parts : the non-lazy part (the `length` in this example) and
  a lazy trigger to get actual device memory because of blob's delay behavor (in this example, it
  is a `Lazy<DeviceData1Seq>`. In this provider, it will be called `Data1Helper.BlobTransposedSeq`
  
But when there are lot of types, it is hard and error-prone to do it manually. GPUHelperProvider 
tends to use Type Provider tech to automatically generate these code.

TransposedSeq is just one usage, there might be more usage, here is what I can image:

- TransposedSeq
- union (without cases that contains fields) can be translate to enum to be used inside kernel


### Framework

To make it possible to generate code by reflection, we need first get a set of .net types. But this
is hard to do with type provider. Type provider accept so called _static parameters_ which can only
be primitive types, such as int, string. A `System.Type` isn't.

The method I use in this project is, you reference the assemblies in your project, then you gives the
namespaces as string splitted by ';'. For example:

```fsharp
#I @"..\..\bin"
#I @"bin\Release"
#r "Test.GPUTypes.Data.Base.dll"
#r "Test.GPUTypes.Data.Derived.dll"
#r "Alea.TypeProviders.Utilities.dll"
#r "Alea.TypeProviders.GPUHelperDirectives.dll"
#r "Alea.TypeProviders.GPUHelperProvider.dll"

[<Literal>]
let namespaces = "Test.GPUTypes.Data.Base;Test.GPUTypes.Data.Derived"

type GPUHelper = Alea.TypeProviders.GPUHelperProvider<namespaces>
```

Then in the generator, I will load all referenced assembly, and filter out the types that hit into the
required namespaces (also check if they have directive attributes):

```fsharp
let assemblies = cfg.ReferencedAssemblies |> Array.map Path.GetFullPath
printfn "Parsing assemblies from:"
for assembly in assemblies do printfn "* %s" assembly
let types =
    assemblies
    |> Array.map Assembly.LoadFile
    |> Array.map (fun assembly -> assembly.GetTypes())
    |> Array.concat
    |> Array.filter (fun ty -> ty.Namespace <> null)
    |> Array.filter (fun ty -> namespaces.Contains(ty.Namespace))
    |> Array.filter (Util.hasAttribute typeof<GenGPUHelperAttribute> true) 
printfn "Got %d types." types.Length
```

After I filter out the types, I start to create a model to simulate the type namespaces. There are two
types in the framework: `Entity` and `Package`. One package represents a namespace node, and it forms
a tree. One entity represents a type, which contains the code to generate helper types. One package 
also contains multiple entities. In this tree structure, we simulated a namespace.

Both `Entity` and `Package` can be subclassed. `Package` has one subclass `RootPackage`. While `Entity`
has many, for different kind of types, such as `IntegerEntity`, `RecordEntity`. You can dump the tree by
calling `RootPackage.Dump()`.

To find a package or entity, you use `Package.FindPackage(path)` and `Package.FindEntity(type)`.

### TODO: 1

Write an test example to show the benifit of using blob. The idea is, if you have many input, malloc
GPU memory once will be more efficient than malloc multiple times. That is the lazy behavor of blob.
We need to write a test (with the test framework <http://www.aleacubase.com/cudalab/performance_test.html>)
to malloc a 100 MB GPU memory. We can do it by 1) malloc 100 MB once; or 2) malloc 10 MB for 10 times; or
3) malloc 1 MB for 100 times. Then we can have proof of value of blob.

### TODO: 2

Now the provider can support situation of this in TransposedSeq case: if a record has an array field, it
will use BlobArrayCompactSeq as its blob and device type (check the test). But there are case that user
can make sure that the array field in a record seq will be euqal length, so we can save some time by use
another type BlobEqualLengthArrayCompactSeq. We should support this case. 

The code shows the idea looks like:

```fsharp
type Data1 =
    {
        Number1 : float
        Number2 : float
        Integer : int
    }

    static member RandomArray(length:int) =
        Array.init<Data1> length (fun i ->
            {
                Number1 = TestUtil.genRandomDouble -10.0 10.0 i
                Number2 = TestUtil.genRandomDouble -20.0 20.0 i
                Integer = TestUtil.genRandomSInt32 -10 10 i
            })

[<Record>]
type DeviceData1Seq =
    {
        Length : int
        Number1s : deviceptr<float>
        Number2s : deviceptr<float>
        Integers : deviceptr<int>
    }

type Blob with
    member this.CreateData1Seq(host:Data1 seq) =
        let host = host |> Seq.toArray
        let length = host.Length
        let number1s = host |> Array.map (fun x -> x.Number1) |> this.CreateArray
        let number2s = host |> Array.map (fun x -> x.Number2) |> this.CreateArray
        let integers = host |> Array.map (fun x -> x.Integer) |> this.CreateArray
        let bmem : Lazy<DeviceData1Seq> =
            Lazy.Create <| fun _ ->
                {
                    Length = length
                    Number1s = number1s.Ptr
                    Number2s = number2s.Ptr
                    Integers = integers.Ptr
                }
        length, bmem

type Data2 =
    {
        Offset1 : float
        Offset2 : int
        EqualLengthArray1 : float[]
        EqualLengthArray2 : int[]
    }

    static member RandomArray(length:int) =
        Array.init<Data2> length (fun i ->
            {
                Offset1 = TestUtil.genRandomDouble -10.0 10.0 i
                Offset2 = TestUtil.genRandomSInt32 -10 10 i
                EqualLengthArray1 = Array.init 100 (TestUtil.genRandomDouble -10.0 10.0)
                EqualLengthArray2 = Array.init 200 (TestUtil.genRandomSInt32 -10 10)
            })
    
[<Record>]
type DeviceData2Seq =
    {
        Length : int
        Offset1s : deviceptr<float>
        Offset2s : deviceptr<int>
        EqualLengthArray1s : BlobEqualLengthArrayCompactSeq<float>
        EqualLengthArray2s : BlobEqualLengthArrayCompactSeq<int>
    }      

type Blob with
    member this.CreateData2Seq(host:Data2 seq) =
        let host = host |> Seq.toArray
        let length = host.Length
        let offset1s = host |> Array.map (fun x -> x.Offset1) |> this.CreateArray
        let offset2s = host |> Array.map (fun x -> x.Offset2) |> this.CreateArray
        let equalLengthArray1s = host |> Array.map (fun x -> x.EqualLengthArray1) |> this.CreateEqualLengthArrayCompactSeq
        let equalLengthArray2s = host |> Array.map (fun x -> x.EqualLengthArray2) |> this.CreateEqualLengthArrayCompactSeq
        let bmem : Lazy<DeviceData2Seq> =
            Lazy.Create <| fun _ ->
                {
                    Length = length
                    Offset1s = offset1s.Ptr
                    Offset2s = offset2s.Ptr
                    EqualLengthArray1s = equalLengthArray1s
                    EqualLengthArray2s = equalLengthArray2s
                }
        length, bmem

type Data3 =
    {
        Offset1 : float
        Offset2 : int
        Array1 : float[]
        Array2 : int[]
    }

    static member RandomArray(length:int) =
        Array.init<Data3> length (fun i ->
            let n1 = TestUtil.genRandomSInt32 20 100 ()
            let n2 = TestUtil.genRandomSInt32 20 100 ()
            {
                Offset1 = TestUtil.genRandomDouble -10.0 10.0 i
                Offset2 = TestUtil.genRandomSInt32 -10 10 i
                Array1 = Array.init n1 (TestUtil.genRandomDouble -10.0 10.0)
                Array2 = Array.init n2 (TestUtil.genRandomSInt32 -10 10)
            })
    
[<Record>]
type DeviceData3Seq =
    {
        Length : int
        Offset1s : deviceptr<float>
        Offset2s : deviceptr<int>
        Array1s : BlobArrayCompactSeq<float>
        Array2s : BlobArrayCompactSeq<int>
    }      

type Blob with
    member this.CreateData3Seq(host:Data3 seq) =
        let host = host |> Seq.toArray
        let length = host.Length
        let offset1s = host |> Array.map (fun x -> x.Offset1) |> this.CreateArray
        let offset2s = host |> Array.map (fun x -> x.Offset2) |> this.CreateArray
        let array1s = host |> Array.map (fun x -> x.Array1) |> this.CreateArrayCompactSeq
        let array2s = host |> Array.map (fun x -> x.Array2) |> this.CreateArrayCompactSeq
        let bmem : Lazy<DeviceData3Seq> =
            Lazy.Create <| fun _ ->
                {
                    Length = length
                    Offset1s = offset1s.Ptr
                    Offset2s = offset2s.Ptr
                    Array1s = array1s
                    Array2s = array2s
                }
        length, bmem

[<Test>]
let ``custom blob``() =
    let template = cuda {
        let! kernel1 =
            <@ fun (output1:deviceptr<float>) (output2:deviceptr<int>) (input:DeviceData1Seq) ->
                let n = input.Length
                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x
                let mutable i = start
                while i < n do
                    output1.[i] <- input.Number1s.[i] + input.Number2s.[i]
                    output2.[i] <- input.Integers.[i] + 1
                    i <- i + stride @>
            |> Compiler.DefineKernel

        let! kernel2 =
            <@ fun (output1:deviceptr<float>) (output2:deviceptr<int>) (input:DeviceData2Seq) ->
                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x
                let mutable i = start
                let n = input.Length
                while i < n do
                    let mutable sum = input.Offset1s.[i]
                    for j = 0 to input.EqualLengthArray1s.Length - 1 do
                        sum <- sum + input.EqualLengthArray1s.[i, j]
                    output1.[i] <- sum

                    let mutable sum = input.Offset2s.[i]
                    for j = 0 to input.EqualLengthArray2s.Length - 1 do
                        sum <- sum + input.EqualLengthArray2s.[i, j]
                    output2.[i] <- sum

                    i <- i + stride @>
            |> Compiler.DefineKernel

        let! kernel3 =
            <@ fun (output1:deviceptr<float>) (output2:deviceptr<int>) (input:DeviceData3Seq) ->
                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x
                let mutable i = start
                let n = input.Length
                while i < n do
                    let mutable sum = input.Offset1s.[i]
                    for j = 0 to input.Array1s.Length(i) - 1 do
                        sum <- sum + input.Array1s.[i, j]
                    output1.[i] <- sum

                    let mutable sum = input.Offset2s.[i]
                    for j = 0 to input.Array2s.Length(i) - 1 do
                        sum <- sum + input.Array2s.[i, j]
                    output2.[i] <- sum

                    i <- i + stride @>
            |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel1 = program.Apply kernel1
            let kernel2 = program.Apply kernel2
            let kernel3 = program.Apply kernel3
            let lp = LaunchParam(32, 256)

            let run (logger:ITimingLogger) =
                use blob = new Blob(worker, logger)

                let input' = Data1.RandomArray(1000)
                let output1' = input' |> Array.map (fun x -> x.Number1 + x.Number2)
                let output2' = input' |> Array.map (fun x -> x.Integer + 1)
                
                let length, input = blob.CreateData1Seq(input')
                let output1 = blob.CreateArray<float>(length)
                let output2 = blob.CreateArray<int>(length)

                kernel1.Launch lp output1.Ptr output2.Ptr input.Value
                TestUtil.assertArrayEqual (Some 1e-7) output1' (output1.Gather())
                TestUtil.assertArrayEqual None output2' (output2.Gather())
                if doprint then TestUtil.testLaunchingTime worker "1" 5000 input.Value

                let input' = Data2.RandomArray(1000)
                let output1' = input' |> Array.map (fun x -> x.Offset1 + (x.EqualLengthArray1 |> Array.sum))
                let output2' = input' |> Array.map (fun x -> x.Offset2 + (x.EqualLengthArray2 |> Array.sum))

                let length, input = blob.CreateData2Seq(input')
                let output1 = blob.CreateArray<float>(length)
                let output2 = blob.CreateArray<int>(length)

                kernel2.Launch lp output1.Ptr output2.Ptr input.Value
                TestUtil.assertArrayEqual (Some 1e-7) output1' (output1.Gather())
                TestUtil.assertArrayEqual None output2' (output2.Gather())
                if doprint then TestUtil.testLaunchingTime worker "2" 5000 input.Value

                let input' = Data3.RandomArray(1000)
                let output1' = input' |> Array.map (fun x -> x.Offset1 + (x.Array1 |> Array.sum))
                let output2' = input' |> Array.map (fun x -> x.Offset2 + (x.Array2 |> Array.sum))

                let length, input = blob.CreateData3Seq(input')
                let output1 = blob.CreateArray<float>(length)
                let output2 = blob.CreateArray<int>(length)

                kernel3.Launch lp output1.Ptr output2.Ptr input.Value
                TestUtil.assertArrayEqual (Some 1e-7) output1' (output1.Gather())
                TestUtil.assertArrayEqual None output2' (output2.Gather())
                if doprint then TestUtil.testLaunchingTime worker "3" 5000 input.Value

            run ) }

    let worker = Worker.Default
    use program = template |> Compiler.load worker
    let logger = TimingLogger("Blob")
    worker.Eval <| fun _ -> program.Run logger
    if doprint then logger.DumpLogs()
```

So, in this example, we have Data1, data2, data3. data1 is simplest case, all field is not array,
data3 is some field has array. both 1 and 3 are supported yet. The data2 case is not done.

To do that, we might add a new directive attribute in GPUHelperDirectives, people can use the
attribute to mark one array field to say it will be equal length. So in generating, we will check
this mark, and then switch the TransposedSeq type and BlobTransposedSeq type.

### TODO : 3

We should code a performance test (using the performance framework) to show the performance acceleration
we got by using transposed data or not use tranposed data.


  






