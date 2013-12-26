namespace Alea.TypeProviders.GPUHelperDirectives

open System

[<AbstractClass>]
type GenerateAttribute() = inherit Attribute()

type TransposedSeqAttribute() = inherit GenerateAttribute()
