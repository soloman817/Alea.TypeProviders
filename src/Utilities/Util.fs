module Alea.TypeProviders.Utilities.Util

open System
open System.Reflection
open Microsoft.FSharp.Core.CompilerServices

type [<Sealed>] AttributeHelper private () =
    
    static member HasAttribute(ty:Type, attrType:Type, isInherit:bool) =
        let attrs = ty.GetCustomAttributes(attrType, isInherit)
        match attrs.Length with
        | 0 -> false
        | _ -> true

    static member HasAttribute(attrs:Attribute seq, attrType:Type) =
        attrs |> Seq.exists (fun attr -> attr.GetType().GUID = attrType.GUID)

let dumpTypeProviderConfig(cfg:TypeProviderConfig) =
    printfn "TypeProviderConfig ==========="
    printfn "IsHostedExecution: %A" cfg.IsHostedExecution
    printfn "IsInvalidationSupported: %A" cfg.IsInvalidationSupported
    printfn "ReferencedAssemblies:"
    for x in cfg.ReferencedAssemblies do
        printfn "  %s" x
    printfn "ResolutionFolder: %s" cfg.ResolutionFolder
    printfn "RuntimeAssembly: %s" cfg.RuntimeAssembly
    printfn "SystemRuntimeAssemblyVersion: %A" cfg.SystemRuntimeAssemblyVersion
    printfn "TemporaryFolder: %s" cfg.TemporaryFolder
    printfn "=============================="
