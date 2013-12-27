namespace Alea.TypeProviders.GPUHelperProvider

open System
open System.IO
open System.Reflection
open System.Collections.Generic
open System.Text.RegularExpressions
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Reflection
open Alea.TypeProviders.Utilities
open Alea.TypeProviders.Utilities.ProvidedTypes
open Alea.TypeProviders.GPUHelperDirectives
open Alea.CUDA
open Alea.CUDA.Utilities

type IntegerEntity(package:Package, ty:Type) =
    inherit Entity(package, ty)

    static let types =
        [ typeof<int8>.GUID     , ( true,  8)
          typeof<int16>.GUID    , ( true, 16)
          typeof<int>.GUID      , ( true, 32)
          typeof<int64>.GUID    , ( true, 64)
          typeof<uint8>.GUID    , (false,  8)
          typeof<uint16>.GUID   , (false, 16)
          typeof<uint32>.GUID   , (false, 32)
          typeof<uint64>.GUID   , (false, 64) ]
        |> Map.ofList

    let signed, bits = types.[ty.GUID]

    static member Register(registry:EntityRegistry) =
        let filter (ty:Type) =
            types.ContainsKey(ty.GUID)

        let create (package:Package) (ty:Type) =
            IntegerEntity(package, ty) :> Entity

        registry.Register(filter, create)

    override this.TransposedSeqType =
        typedefof<deviceptr<_>>.MakeGenericType([| ty |])

    override this.BlobTransposedSeqType =
        typedefof<BlobArray<_>>.MakeGenericType([| ty |])

    override this.InvokeBlobTransposedSeqCreate(blobExpr, hostExpr) =
        let mtds =
            blobExpr.Type.GetMethods()
            |> Array.filter (fun mtd -> mtd.Name = "CreateArray")
            |> Array.filter (fun mtd -> mtd.GetParameters().Length = 1)
            |> Array.filter (fun mtd -> mtd.GetParameters().[0].ParameterType.IsArray)
        let mtd = mtds.[0].MakeGenericMethod([| ty |])
        Expr.Call(blobExpr, mtd, hostExpr :: [])
       
    override this.InvokeBlobTransposedSeqTrigger(bmemExpr) =
        Expr.PropertyGet(bmemExpr, bmemExpr.Type.GetProperty("Ptr"))

    override this.ToString() =
        let name = if signed then "int" else "uint"
        sprintf "%s%d" name bits

    override this.Dump(indent) =
        let indentstr = String.replicate indent "."
        printfn "%s* %s [%O]" indentstr this.Name this

type FloatingPointEntity(package:Package, ty:Type) =
    inherit Entity(package, ty)

    static let types =
        [ typeof<float32>.GUID  , 32
          typeof<float>.GUID    , 64 ]
        |> Map.ofList

    let bits = types.[ty.GUID]

    static member Register(registry:EntityRegistry) =
        let filter (ty:Type) =
            types.ContainsKey(ty.GUID)

        let create (package:Package) (ty:Type) =
            FloatingPointEntity(package, ty) :> Entity

        registry.Register(filter, create)

    override this.TransposedSeqType =
        typedefof<deviceptr<_>>.MakeGenericType([| ty |])
        
    override this.BlobTransposedSeqType =
        typedefof<BlobArray<_>>.MakeGenericType([| ty |])

    override this.InvokeBlobTransposedSeqCreate(blobExpr, hostExpr) =
        let mtds =
            blobExpr.Type.GetMethods()
            |> Array.filter (fun mtd -> mtd.Name = "CreateArray")
            |> Array.filter (fun mtd -> mtd.GetParameters().Length = 1)
            |> Array.filter (fun mtd -> mtd.GetParameters().[0].ParameterType.IsArray)
        let mtd = mtds.[0].MakeGenericMethod([| ty |])
        Expr.Call(blobExpr, mtd, hostExpr :: [])
       
    override this.InvokeBlobTransposedSeqTrigger(bmemExpr) =
        Expr.PropertyGet(bmemExpr, bmemExpr.Type.GetProperty("Ptr"))
       
    override this.ToString() =
        sprintf "float%d" bits

    override this.Dump(indent) =
        let indentstr = String.replicate indent "."
        printfn "%s* %s (%O)" indentstr this.Name this

type ArrayEntity(package:Package, ty:Type) =
    inherit Entity(package, ty)

    let elementType = ty.GetElementType()

    static member Register(registry:EntityRegistry) =
        let filter (ty:Type) =
            ty.IsArray && ty.GetArrayRank() = 1

        let create (package:Package) (ty:Type) =
            ArrayEntity(package, ty) :> Entity

        registry.Register(filter, create)

    override this.TransposedSeqType =
        typedefof<BlobArrayCompactSeq<_>>.MakeGenericType([| elementType |])

    override this.BlobTransposedSeqType =
        typedefof<BlobArrayCompactSeq<_>>.MakeGenericType([| elementType |])

    override this.InvokeBlobTransposedSeqCreate(blobExpr, hostExpr) =
        let mtd = blobExpr.Type.GetMethod("CreateArrayCompactSeq").MakeGenericMethod([| elementType |])
        Expr.Call(blobExpr, mtd, hostExpr :: [])

    override this.InvokeBlobTransposedSeqTrigger(bmemExpr) =
        bmemExpr

type RecordEntity(package:Package, ty:Type) as this =
    inherit Entity(package, ty)

    let fields = FSharpType.GetRecordFields(ty) |> List.ofArray

    let providedHostType = Lazy.Create <| fun _ ->
        printfn "Generating EntityHostType (%s.%s) ..." this.Path this.Name
        let hostType = ProvidedTypeDefinition(sprintf "%sHelper" this.Name, Some typeof<obj>, IsErased = false)
        package.ProvidedHostType.AddMember hostType
        hostType

    let invokeBlobTransposedSeqCreate (bseqType:Type) (blobExpr:Expr) (hostExpr:Expr) =
        // inputs: blob:Blob, host:'T[]
        let lengthExpr = Expr.PropertyGet(hostExpr, hostExpr.Type.GetProperty("Length"))
        let arrayMapMethod = typeof<unit>.Assembly.GetType("Microsoft.FSharp.Collections.ArrayModule").GetMethod("Map")

        // (fun x -> x.Property)
        let lambdaExprs = fields |> List.map (fun field ->
            let var = Var("x", ty)
            let bodyExpr = Expr.PropertyGet(Expr.Var(var), ty.GetProperty(field.Name))
            Expr.Lambda(var, bodyExpr))

        // Array.map (fun x -> x.Property) host
        let transposedExprs = (fields, lambdaExprs) ||> List.map2 (fun field lambdaExpr ->
            let arrayMapMethod = arrayMapMethod.MakeGenericMethod([| ty; field.PropertyType |])
            Expr.Call(arrayMapMethod, lambdaExpr :: hostExpr :: []))

        // blob.CreateXXX(Array.map (fun x -> x.Property) host) : obj
        let bmemExprs = (fields, transposedExprs) ||> List.map2 (fun field transposedExpr ->
            let entity = package.FindEntity(field.PropertyType)
            entity.InvokeBlobTransposedSeqCreate(blobExpr, transposedExpr))

        // bseq.ctor(length, bmem1, bmem2, ...)
        let bseqCtor = bseqType.GetConstructors().[0]
        Expr.NewObject(bseqCtor, lengthExpr :: bmemExprs)

    let invokeBlobTransposedSeqTrigger (dseqType:Type) (bmemExpr:Expr) =
        let lengthExpr = Expr.PropertyGet(bmemExpr, bmemExpr.Type.GetProperty("Length"))
        let deviceExprs = fields |> List.map (fun field ->
            let bmemExpr = Expr.PropertyGet(bmemExpr, bmemExpr.Type.GetProperty(field.Name))
            let entity = package.FindEntity(field.PropertyType)
            entity.InvokeBlobTransposedSeqTrigger(bmemExpr))
        let dseqCtor = dseqType.GetConstructors().[0]
        Expr.NewObject(dseqCtor, lengthExpr :: deviceExprs)

    let providedTransposedSeqType = Lazy.Create <| fun _ ->
        printfn "Generating TransposedSeqType (%s.%s) ..." this.Path this.Name
        let providedTransposedSeqType = ProvidedTypeDefinition("TransposedSeq", Some typeof<TransposeSeq>, IsErased = false)
        providedHostType.Value.AddMember providedTransposedSeqType

        // fields
        let fields = fields |> List.map (fun field ->
            let fieldName = sprintf "_%s" field.Name
            let fieldType = package.FindEntity(field.PropertyType).TransposedSeqType
            ProvidedField(fieldName, fieldType))
        let fields = ProvidedField("_Length", typeof<int>) :: fields
        providedTransposedSeqType.AddMembers fields

        // constructor
        let ctor =
            let parameters = fields |> List.map (fun field -> ProvidedParameter(field.Name, field.FieldType))
            let invokeCode (args:Expr list) =
                match args with
                | thisExpr :: valueExprs ->
                    (fields, valueExprs) ||> List.fold2 (fun firstExpr field valueExpr ->
                        let secondExpr = Expr.FieldSet(thisExpr, field, valueExpr)
                        Expr.Sequential(firstExpr, secondExpr)) thisExpr
                | _ -> failwith "Won't happen."
            ProvidedConstructor(parameters, InvokeCode = invokeCode)
        providedTransposedSeqType.AddMember ctor

        // properties
        fields |> List.iter (fun field ->
            let propertyName = field.Name.Substring(1) // remove "_" prefix
            let propertyType = field.FieldType
            let getterCode (args:Expr list) = Expr.FieldGet(args.[0], field)
            let property = ProvidedProperty(propertyName, propertyType, GetterCode = getterCode)
            providedTransposedSeqType.AddMember property)

        providedTransposedSeqType

    let providedBlobTransposedSeqType = Lazy.Create <| fun _ ->
        printfn "Generating BlobTransposedSeqType (%s.%s) ..." this.Path this.Name
        let providedBlobTransposedSeqType = ProvidedTypeDefinition("BlobTransposedSeq", Some typeof<obj>, IsErased = false)
        providedHostType.Value.AddMember providedBlobTransposedSeqType

        // fields
        let fields = fields |> List.map (fun field ->
            let fieldName = sprintf "_%s" field.Name
            let fieldType = package.FindEntity(field.PropertyType).BlobTransposedSeqType
            ProvidedField(fieldName, fieldType))
        let fields = ProvidedField("_Length", typeof<int>) :: fields
        providedBlobTransposedSeqType.AddMembers fields

        // constructor
        let ctor =
            let parameters = fields |> List.map (fun field -> ProvidedParameter(field.Name, field.FieldType))
            let invokeCode (args:Expr list) =
                match args with
                | thisExpr :: valueExprs ->
                    (fields, valueExprs) ||> List.fold2 (fun firstExpr field valueExpr ->
                        let secondExpr = Expr.FieldSet(thisExpr, field, valueExpr)
                        Expr.Sequential(firstExpr, secondExpr)) thisExpr
                | _ -> failwith "Won't happen."
            ProvidedConstructor(parameters, InvokeCode = invokeCode)
        providedBlobTransposedSeqType.AddMember ctor

        // properties
        fields |> List.iter (fun field ->
            let propertyName = field.Name.Substring(1) // remove "_" prefix
            let propertyType = field.FieldType
            let getterCode (args:Expr list) = Expr.FieldGet(args.[0], field)
            let property = ProvidedProperty(propertyName, propertyType, GetterCode = getterCode)
            providedBlobTransposedSeqType.AddMember property)

        // static member Create(blob:Blob, host:'T[]) : (bmem:'BlobTSeq)
        let mtd =
            let returnType = providedBlobTransposedSeqType
            let hostType = ty.MakeArrayType()
            let parameters = ProvidedParameter("blob", typeof<Alea.CUDA.Utilities.Blob.Blob>) :: ProvidedParameter("host", hostType) :: []
            let invokeCode (args:Expr list) = invokeBlobTransposedSeqCreate providedBlobTransposedSeqType args.[0] args.[1]
            ProvidedMethod("Create", parameters, returnType, IsStaticMethod = true, InvokeCode = invokeCode)
        providedBlobTransposedSeqType.AddMember mtd

        // member this.Device : (device:'TSeq)
        let property = 
            let propertyType = providedTransposedSeqType.Value
            let getterCode (args:Expr list) = invokeBlobTransposedSeqTrigger propertyType args.[0]
            ProvidedProperty("Device", propertyType, GetterCode = getterCode)
        providedBlobTransposedSeqType.AddMember property

        providedBlobTransposedSeqType

    static member Register(registry:EntityRegistry) =
        let filter (ty:Type) =
            FSharpType.IsRecord(ty)

        let create (package:Package) (ty:Type) =
            RecordEntity(package, ty) :> Entity

        registry.Register(filter, create)

    override this.TransposedSeqType = providedTransposedSeqType.Value :> Type
    override this.BlobTransposedSeqType = providedBlobTransposedSeqType.Value :> Type
    override this.InvokeBlobTransposedSeqCreate(blobExpr, hostExpr) = invokeBlobTransposedSeqCreate this.BlobTransposedSeqType blobExpr hostExpr
    override this.InvokeBlobTransposedSeqTrigger(bmemExpr) = invokeBlobTransposedSeqTrigger this.TransposedSeqType bmemExpr

    override this.Initialize() =
        fields |> List.iter (fun field -> package.InstallEntity(field.PropertyType))

    override this.ToString() =
        sprintf "Record(%d fields)" fields.Length

    override this.Dump(indent) =
        let indentstr = String.replicate indent "."
        printfn "%s* %s (%O)" indentstr this.Name this

    override this.Generate() =
        if Util.hasAttribute typeof<GenGPUHelperAttribute> false ty then
            providedTransposedSeqType.Value |> ignore
            providedBlobTransposedSeqType.Value |> ignore
