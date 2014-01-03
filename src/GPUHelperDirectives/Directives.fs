namespace Alea.TypeProviders.GPUHelperDirectives

open System

[<AttributeUsage(AttributeTargets.All, AllowMultiple = false)>]
type GenGPUHelperAttribute() = inherit Attribute()

[<AttributeUsage(AttributeTargets.All, AllowMultiple = true)>]
type GenGPUHelperByAttribute(tys:Type[]) =
    inherit GenGPUHelperAttribute()
    member this.TypeArguments = tys

type EqualLengthArrayAttribute() = inherit Attribute()