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

    override this.InvokeTransposedSeqTypeBlobCreate(blobExpr, hostExpr) =
        let blobType = blobExpr.Type
        let methodInfos = blobType.GetMethods()
        let methodInfos = methodInfos |> Array.filter (fun info -> info.Name = "CreateArray")
        let methodInfos = methodInfos |> Array.filter (fun (info:MethodInfo) ->
            let parameters = info.GetParameters()
            parameters.[0].ParameterType.IsArray)
        let methodInfo = methodInfos.[0]
        let methodInfo = methodInfo.MakeGenericMethod([| ty |])
        Expr.Coerce(Expr.Call(blobExpr, methodInfo, hostExpr :: []), typeof<obj>)
       
    override this.InvokeTransposedSeqTypeBlobTrigger(bmemExpr) =
        let bmemType = typedefof<BlobArray<_>>
        let bmemType = bmemType.MakeGenericType([| ty |])
        let bmemExpr = Expr.Coerce(bmemExpr, bmemType)
        Expr.PropertyGet(bmemExpr, bmemType.GetProperty("Ptr"))

    override this.ToString() =
        let name = if signed then "int" else "uint"
        sprintf "%s%d" name bits

    override this.Dump(indent) =
        let indentstr = String.replicate indent " "
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
        
    override this.InvokeTransposedSeqTypeBlobCreate(blobExpr, hostExpr) =
        let blobType = blobExpr.Type
        let methodInfos = blobType.GetMethods()
        let methodInfos = methodInfos |> Array.filter (fun info -> info.Name = "CreateArray")
        let methodInfos = methodInfos |> Array.filter (fun (info:MethodInfo) ->
            let parameters = info.GetParameters()
            parameters.[0].ParameterType.IsArray)
        let methodInfo = methodInfos.[0]
        let methodInfo = methodInfo.MakeGenericMethod([| ty |])
        Expr.Coerce(Expr.Call(blobExpr, methodInfo, hostExpr :: []), typeof<obj>)

    override this.InvokeTransposedSeqTypeBlobTrigger(bmemExpr) =
        let bmemType = typedefof<BlobArray<_>>
        let bmemType = bmemType.MakeGenericType([| ty |])
        let bmemExpr = Expr.Coerce(bmemExpr, bmemType)
        Expr.PropertyGet(bmemExpr, bmemType.GetProperty("Ptr"))
       
    override this.ToString() =
        sprintf "float%d" bits

    override this.Dump(indent) =
        let indentstr = String.replicate indent " "
        printfn "%s* %s (%O)" indentstr this.Name this

type RecordEntity(package:Package, ty:Type) as this =
    inherit Entity(package, ty)

    let fields = FSharpType.GetRecordFields(ty) |> List.ofArray

    let providedHostType = Lazy.Create <| fun _ ->
        printfn "Generating EntityHostType (%s.%s) ..." this.Path this.Name
        let hostType = ProvidedTypeDefinition(sprintf "%sHelper" this.Name, Some typeof<obj>, IsErased = false)
        package.ProvidedHostType.AddMember hostType
        hostType

    let invokeTransposedSeqTypeBlobCreate (blobExpr:Expr) (hostExpr:Expr) =
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
            let bmemExpr = entity.InvokeTransposedSeqTypeBlobCreate(blobExpr, transposedExpr)
            if bmemExpr.Type <> typeof<obj> then failwith "BlobMemory type must be typeof<obj>."
            bmemExpr)

        let bmemsExpr = Expr.NewArray(typeof<obj>, bmemExprs)
        let bmemType = typeof<TransposedSeqBlobMemory>
        let bmemCtor = bmemType.GetConstructor([| typeof<int>; typeof<obj[]> |])
        Expr.NewObject(bmemCtor, lengthExpr :: bmemsExpr :: [])

    let invokeTransposedSeqTypeBlobTrigger (seqType:Type) (bmemExpr:Expr) =
        if bmemExpr.Type <> typeof<TransposedSeqBlobMemory> then failwith "BlobMemory type must be typeof<TransposedSeqBlobMemory>."
        let lengthExpr = Expr.PropertyGet(bmemExpr, bmemExpr.Type.GetProperty("Length"))
        let bmemsExpr = Expr.PropertyGet(bmemExpr, bmemExpr.Type.GetProperty("BlobMemories"))
        let deviceExprs = fields |> List.mapi (fun i field ->
            let entity = package.FindEntity(field.PropertyType)
            entity.InvokeTransposedSeqTypeBlobTrigger(<@@ (%%bmemsExpr:obj[]).[i] @@>))
        let ctor = seqType.GetConstructors().[0]
        Expr.NewObject(ctor, lengthExpr :: deviceExprs)

    let providedTransposedSeqType = Lazy.Create <| fun _ ->
        printfn "Generating TransposedSeqType (%s.%s) ..." this.Path this.Name
        let transposedSeqType = ProvidedTypeDefinition("TransposedSeq", Some typeof<TransposeSeq>, IsErased = false)
        providedHostType.Value.AddMember transposedSeqType

        // fields
        let fields = fields |> List.map (fun field ->
            let fieldName = sprintf "_%s" field.Name
            let fieldType = package.FindEntity(field.PropertyType).TransposedSeqType
            ProvidedField(fieldName, fieldType))
        let fields = ProvidedField("_Length", typeof<int>) :: fields
        transposedSeqType.AddMembers fields

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
        transposedSeqType.AddMember ctor

        // properties
        fields |> List.iter (fun field ->
            let propertyName = field.Name.Substring(1) // remove "_" prefix
            let propertyType = field.FieldType
            let getterCode (args:Expr list) = Expr.FieldGet(args.[0], field)
            let property = ProvidedProperty(propertyName, propertyType, GetterCode = getterCode)
            transposedSeqType.AddMember property)

        // method BlobCreate(blob:Blob, host:'T[]) : (bmem:TransposedSeqBlobMemory)
        let mtd =
            let returnType = typeof<TransposedSeqBlobMemory>
            let hostType = ty.MakeArrayType()
            let parameters = ProvidedParameter("blob", typeof<Alea.CUDA.Utilities.Blob.Blob>) :: ProvidedParameter("host", hostType) :: []
            let invokeCode (args:Expr list) = invokeTransposedSeqTypeBlobCreate args.[0] args.[1]
            ProvidedMethod("BlobCreate", parameters, returnType, IsStaticMethod = true, InvokeCode = invokeCode)
        transposedSeqType.AddMember mtd

        // method BlobTrigger(bmem:TransposedSeqBlobMemory) : (device:'TSeq)
        let mtd = 
            let returnType = transposedSeqType
            let bmemType = typeof<TransposedSeqBlobMemory>
            let parameters = ProvidedParameter("bmem", bmemType) :: []
            let invokeCode (args:Expr list) = invokeTransposedSeqTypeBlobTrigger transposedSeqType args.[0]
            ProvidedMethod("BlobTrigger", parameters, returnType, IsStaticMethod = true, InvokeCode = invokeCode)
        transposedSeqType.AddMember mtd

        transposedSeqType

    static member Register(registry:EntityRegistry) =
        let filter (ty:Type) =
            FSharpType.IsRecord(ty)

        let create (package:Package) (ty:Type) =
            RecordEntity(package, ty) :> Entity

        registry.Register(filter, create)

    override this.TransposedSeqType = providedTransposedSeqType.Value :> Type
    override this.InvokeTransposedSeqTypeBlobCreate(blobExpr, hostExpr) = Expr.Coerce(invokeTransposedSeqTypeBlobCreate blobExpr hostExpr, typeof<obj>)
    override this.InvokeTransposedSeqTypeBlobTrigger(bmemExpr) = Expr.Coerce(bmemExpr, typeof<TransposedSeqBlobMemory>) |> invokeTransposedSeqTypeBlobTrigger this.TransposedSeqType

    override this.Initialize() =
        fields |> List.iter (fun field -> package.InstallEntity(field.PropertyType))

    override this.ToString() =
        sprintf "Record(%d fields)" fields.Length

    override this.Dump(indent) =
        let indentstr = String.replicate indent " "
        printfn "%s* %s (%O)" indentstr this.Name this

    override this.Generate() =
        if Util.hasAttribute typeof<TransposedSeqAttribute> false ty then
            providedTransposedSeqType.Value |> ignore
