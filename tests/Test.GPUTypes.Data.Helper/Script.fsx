#I @"..\..\bin"
#I @"bin\Release"
#r "TestData.MyGPUTypes.Base.dll"
#r "TestData.MyGPUTypes.Derived.dll"
#r "Alea.TypeProviders.Utilities.dll"
#r "Alea.TypeProviders.GPUHelperDirectives.dll"
#r "Alea.TypeProviders.GPUHelperProvider.dll"

[<Literal>]
let namespaces = "TestData.MyGPUTypes.Base;TestData.MyGPUTypes.Derived"

type GPUHelper = Alea.TypeProviders.GPUHelperProvider<namespaces>



