namespace Alea.TypeProviders.GPUHelperDirectives

open System
open System.IO
open System.Reflection
open System.Collections.Generic
open System.Text.RegularExpressions
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities

type GPUObjectAttribute() =
    inherit Attribute()

    interface ICustomTypeBuilder with
        member this.Build(ctx, clrType) =
            let irFields =
                clrType.GetFields(BindingFlags.NonPublic ||| BindingFlags.Instance)
                |> Array.map (fun info ->
                    let irFieldName = info.Name.Substring(1)
                    let irFieldType = IRTypeBuilder.Instance.Build(ctx, info.FieldType)
                    irFieldName, irFieldType)

            let param = IRStructOrUnionBuildingParam.Create(irFields)
            let irStructType = IRStructType.Create(ctx.IRContext, param, IRRefTypeHint.Default)
            irStructType |> Some

    interface ICustomToUnmanagedMarshaler with
        member this.Marshal(irType, clrObject, buffer) =
            let clrType = clrObject.GetType()
            let irFieldInfos = irType.Struct.FieldInfos
            let irFieldLayouts = irType.Struct.FieldLayouts

            (irFieldInfos, irFieldLayouts) ||> Array.iter2 (fun irFieldInfo irFieldLayout ->
                let clrField = clrType.GetProperty(irFieldInfo.Name).GetGetMethod().Invoke(clrObject, Array.empty)
                let buffer = buffer + (irFieldLayout.Offset |> nativeint)
                ToUnmanagedMarshaler.Instance.Marshal(irFieldInfo.FieldType, clrField, buffer))

            Some()

[<AbstractClass;GPUObject>]
type GPUObject() = class end

type TransposeSeq() = inherit GPUObject()

