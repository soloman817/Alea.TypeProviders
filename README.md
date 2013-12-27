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
- A blob memory type which has two parts : the non-lazy part (the length in this example) and
  a lazy trigger to get actual device memory because of blob's delay behavor (in this example, it
  is a Lazy<DeviceData1Seq>. In this provider, it will be called `Data1Helper.BlobTransposedSeq`
  
But when there are lot of types, it is hard and error-prone to do it manually. GPUHelperProvider 
tends to use Type Provider tech to automatically generate these code.

TransposedSeq is just one usage, there might be more usage, here is what I can image:

- TransposedSeq
- union (without cases that contains fields) can be translate to enum to be used inside kernel


### Framework

To make it possible to generate code by reflection, we need first get a set of .net types. But this
is hard to do with type provider. Type provider accept so called _static parameters_ which can only
be primitive types, such as int, string. A `System.Type` isn't. So






