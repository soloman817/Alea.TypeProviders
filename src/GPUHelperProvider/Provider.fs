namespace Alea.TypeProviders.GPUHelperProvider

open System
open System.IO
open System.Reflection
open System.Collections.Generic
open System.Text.RegularExpressions
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Reflection
open Microsoft.FSharp.Core.CompilerServices
open Alea.TypeProviders.Utilities
open Alea.TypeProviders.Utilities.ProvidedTypes
open Alea.TypeProviders.GPUHelperDirectives

[<assembly: TypeProviderAssembly>]
do ()

type Helper(thisAssembly:Assembly, providedNamespace:string, name:string, namespaces:Set<string>, cfg:TypeProviderConfig) =
    let debug = false
    let mutable optDllPath : string option = None
    let mutable optProvidedType : ProvidedTypeDefinition option = None

    member this.Name = name
    member this.Namespaces = namespaces
    member this.ProvidedType = optProvidedType.Value

    member this.Generate() =
        if optProvidedType.IsNone then
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

            let registry = EntityRegistry()
            IntegerEntity.Register(registry)
            FloatingPointEntity.Register(registry)
            ArrayEntity.Register(registry)
            RecordEntity.Register(registry)

            optDllPath <- debug |> function
                | true  -> @"C:\Users\Xiang\Desktop\AAA.dll" |> Some
                | false -> Path.ChangeExtension(Path.GetTempFileName(), ".dll") |> Some
            let dllPath = optDllPath.Value
            let providedAssembly = ProvidedAssembly(dllPath)
            let hostType = ProvidedTypeDefinition(thisAssembly, providedNamespace, name, Some typeof<obj>, IsErased = false, HideObjectMethods = true)
            providedAssembly.AddTypes(hostType :: [])

            let rootPackage = RootPackage(name, registry, hostType)
            types |> Array.iter rootPackage.InstallEntity
            rootPackage.Dump()
            printfn "Generating RootPackageHostType (%s) ..." name
            rootPackage.Entities |> Seq.iter (fun entity -> entity.Generate())
            optProvidedType <- rootPackage.ProvidedHostType |> Some

    member this.Dispose(disposing:bool) =
        if not debug then optDllPath |> Option.iter (fun dllPath -> try File.Delete(dllPath) with _ -> ())

    member this.Dispose() = this.Dispose(true); GC.SuppressFinalize(this)
    override this.Finalize() = this.Dispose(false)
    interface IDisposable with member this.Dispose() = this.Dispose()

[<TypeProvider>]
type Provider(cfg:TypeProviderConfig) as this =
    inherit TypeProviderForNamespaces()

    //do Util.dumpTypeProviderConfig cfg

    let thisAssembly = Assembly.GetExecutingAssembly()
    let providedNamespace = "Alea.TypeProviders"
    let helpers = Dictionary<string, Helper>()

    let generateHelper (name:string) (parameters:obj[]) =
        if helpers.ContainsKey(name) then helpers.[name].ProvidedType
        else
            let namespaces = parameters.[0] :?> string
            let namespaces = Regex.Split(namespaces, ";") |> Set.ofArray
            let helper = new Helper(thisAssembly, providedNamespace, name, namespaces, cfg)
            helper.Generate()
            helpers.Add(name, helper)
            helper.ProvidedType

    let helperProvider =
        let ty = ProvidedTypeDefinition(thisAssembly, providedNamespace, "GPUHelperProvider", Some typeof<obj>, IsErased = false)
        let parameters = ProvidedStaticParameter("Namespaces", typeof<string>) :: []
        ty.DefineStaticParameters(parameters, generateHelper)
        ty

    do System.AppDomain.CurrentDomain.add_AssemblyResolve(fun _ args ->
        printfn "Resolving %s ... " args.Name
        let name = System.Reflection.AssemblyName(args.Name)
        let existingAssembly = 
            System.AppDomain.CurrentDomain.GetAssemblies()
            |> Seq.tryFind(fun a -> System.Reflection.AssemblyName.ReferenceMatchesDefinition(name, a.GetName()))
        match existingAssembly with
        | Some a -> a
        | None -> null
        )

    do this.AddNamespace(providedNamespace, helperProvider :: [])

    member this.Dispose(disposing:bool) =
        if disposing then helpers.Values |> Seq.iter (fun helper -> helper.Dispose())

    member this.Dispose() = this.Dispose(true); GC.SuppressFinalize(this)
    override this.Finalize() = this.Dispose(false)
    interface IDisposable with member this.Dispose() = this.Dispose()
