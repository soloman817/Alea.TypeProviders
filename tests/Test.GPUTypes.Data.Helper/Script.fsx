#I @"..\..\bin"
#I @"bin\Release"
#r "Test.GPUTypes.Data.Base.dll"
#r "Test.GPUTypes.Data.Derived.dll"
#r "Alea.TypeProviders.Utilities.dll"
#r "Alea.TypeProviders.GPUHelperDirectives.dll"
#r "Alea.TypeProviders.GPUHelperProvider.dll"

[<Literal>]
let namespaces = "Test.GPUTypes.Data.Base;Test.GPUTypes.Data.Derived"

type GPUHelper = Alea.TypeProviders.GPUHelperProvider<namespaces>



