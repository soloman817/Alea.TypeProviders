namespace Alea.TypeProviders.GPUHelperProvider

open System
open System.IO
open System.Reflection
open System.Collections.Generic
open System.Text.RegularExpressions
open Microsoft.FSharp.Quotations
open Alea.TypeProviders.Utilities
open Alea.TypeProviders.Utilities.ProvidedTypes
open Alea.TypeProviders.GPUHelperDirectives

type [<Sealed>] EntityRegistry() =
    let registry = List<(Type -> bool) * (Package -> Type -> Entity)>()

    member this.Register(filter, create) = registry.Add(filter, create)

    member this.IsTypeSupported(ty:Type) =
        registry |> Seq.map fst |> Seq.exists (fun filter -> filter ty)

    member this.IsTypeNotSupported(ty:Type) =
        not <| this.IsTypeSupported(ty)

    member this.TryCreateEntity(package:Package, ty:Type) =
        registry |> Seq.tryPick (fun (filter, create) -> filter ty |> function
            | true -> create package ty |> Some
            | false -> None)

    member this.CreateEntity(package:Package, ty:Type) =
        this.TryCreateEntity(package, ty) |> function
        | Some entity -> entity
        | None -> failwithf "Not supported type %O" ty

and [<AbstractClass>] Entity(package:Package, ty:Type) =
    member this.Package = package
    member this.Type = ty

    override this.ToString() = sprintf "[%s]" ty.FullName

    static member NameOf(package:Package, ty:Type) =
        match ty.IsGenericType with
        | false -> ty.Name
        | true ->
            let name = ty.Name.Substring(0, ty.Name.IndexOf("`"))
            ty.GetGenericArguments()
            |> Array.map (fun ty ->
                let entity : Entity = package.FindEntity(ty)
                let name : string = entity.Name
                name.Substring(0, 1).ToUpper() + name.Substring(1))
            |> String.concat "And"
            |> sprintf "%sBy%s" name

    member this.Name = Entity.NameOf(package, ty)

    member this.Path = package.Path

    abstract Dump : int -> unit
    default this.Dump(ident) =
        let indentstr = String.replicate ident "."
        printfn "%s* %s" indentstr this.Name

    abstract Initialize : unit -> unit
    default this.Initialize() = ()

    abstract Generate : unit -> unit
    default this.Generate() = ()

    abstract GetTransposedSeqType : Attribute seq -> Type // 'TranSeq
    default this.GetTransposedSeqType(_) = failwithf "%O.GetTransposedSeqType not implemented." this

    abstract GetBlobTransposedSeqType : Attribute seq -> Type // 'BlobTranSeq
    default this.GetBlobTransposedSeqType(_) = failwithf "%O.BlobTransposedSeqType not supported." this

    abstract InvokeBlobTransposedSeqCreate : Type * Expr * Expr -> Expr  // (blobExpr:Blob, hostExpr:'T[]) -> (bmemExpr:'BlobTranSeq)
    default this.InvokeBlobTransposedSeqCreate(_,_,_) = failwithf "%O.InvokeBlobTransposedSeqCreate not supported." this

    abstract InvokeBlobTransposedSeqTrigger : Type * Expr -> Expr // (bmemExpr:'BlobTranSeq) -> (deviceExpr:'TranSeq)
    default this.InvokeBlobTransposedSeqTrigger(_,_) = failwithf "%O.InvokeBlobTransposedSeqTrigger not supported." this

and Package(parent:Package option, name:string) as this =
    let children = Dictionary<string, Package>()
    let entities = Dictionary<string, Entity>()

    let providedHostType = Lazy.Create <| fun _ -> 
        printfn "Generating PackageHostType (%s) ..." this.Path
        let hostType = ProvidedTypeDefinition(name, Some typeof<obj>, IsErased = false)
        parent |> Option.iter (fun parent -> parent.ProvidedHostType.AddMember hostType)
        hostType

    member this.IsRoot = parent.IsNone
    member this.Parent = match parent with Some parent -> parent | None -> failwith "No parent package."
    member this.Root = parent |> function Some package -> package.Root | None -> this
    member this.Name = name

    member this.Path = this.IsRoot |> function
        | true -> ""
        | false ->
            let parentPath = this.Parent.Path
            if parentPath = "" then name
            else sprintf "%s.%s" parentPath name

    override this.ToString() = sprintf "[%s]" this.Path

    member private this.ChildrenEntities =
        let children = children.Values
        let entities = entities.Values
        seq {
            yield! entities
            for child in children do
                yield! child.ChildrenEntities }

    member this.Entities = this.Root.ChildrenEntities

    member private this.GetPackage(path:string list) =
        let getMyPackage(name:string) =
            if children.ContainsKey(name) then children.[name]
            else
                let package = Package(Some this, name)
                children.Add(name, package)
                package

        match path with
        | [] -> this
        | name :: [] -> getMyPackage(name)
        | name :: restPath -> getMyPackage(name).GetPackage(restPath)

    member this.FindPackage(path:string) =
        let path = Regex.Split(path, @"\.") |> List.ofArray
        this.Root.GetPackage(path)

    abstract EntityRegistry : EntityRegistry
    default this.EntityRegistry = this.Root.EntityRegistry

    member private this.GetEntity(ty:Type) =
        if ty.Namespace <> this.Path then failwith "Namespace not match path."
        let tyn = Entity.NameOf(this, ty)
        if entities.ContainsKey(tyn) then entities.[tyn]
        else
            let entity = this.EntityRegistry.CreateEntity(this, ty)
            entities.Add(tyn, entity)
            entity.Initialize()
            entity

    member this.FindEntity(ty:Type) =
        this.FindPackage(ty.Namespace).GetEntity(ty)

    member this.InstallEntity(ty:Type) =
        if ty.IsGenericTypeDefinition then
            ty.GetCustomAttributes(typeof<GenGPUHelperByAttribute>, false)
            |> Array.map (fun attr -> (attr :?> GenGPUHelperByAttribute).TypeArguments)
            |> Array.map (fun typeArguments -> ty.MakeGenericType(typeArguments))
            |> Array.iter (fun ty -> this.InstallEntity(ty))
        else
            //printfn "Installing %O ... " ty
            this.FindEntity(ty) |> ignore

    member this.Dump(?indent:int) =
        let indent = defaultArg indent 0
        let indentstr = String.replicate indent "."
        printfn "%s%s" indentstr this.Name
        entities.Values |> Seq.iter (fun entity -> entity.Dump(indent + 1))
        children.Values |> Seq.iter (fun package -> package.Dump(indent + 1))

    abstract ProvidedHostType : ProvidedTypeDefinition
    default this.ProvidedHostType = providedHostType.Value

type [<Sealed>] RootPackage(name:string, registry:EntityRegistry, hostType:ProvidedTypeDefinition) =
    inherit Package(None, name)
    override this.EntityRegistry = registry
    override this.ProvidedHostType = hostType

