module Alea.TypeProviders.Utilities.Util

open System
open System.Reflection
open Microsoft.FSharp.Core.CompilerServices

let hasAttribute (attrType:Type) (isInherit:bool) (ty:Type) =
    let attrs = ty.GetCustomAttributes(attrType, isInherit)
    match attrs.Length with
    | 0 -> false
    | _ -> true

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
