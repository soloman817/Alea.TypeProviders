copy /Y ..\..\..\..\bin\* .
copy /Y Generate.txt Generate.fs

"C:\Program Files (x86)\Microsoft SDKs\F#\3.0\Framework\v4.0\fsc.exe"^
 -o:Test.GPUTypes.Data.Helper.dll^
 --noframework^
 --doc:Test.GPUTypes.Data.Helper.XML^
 -r:FSharp.Core.dll^
 -r:Alea.CUDA.dll^
 -r:Test.GPUTypes.Data.Base.dll^
 -r:Test.GPUTypes.Data.Derived.dll^
 -r:Alea.TypeProviders.Utilities.dll^
 -r:Alea.TypeProviders.GPUHelperDirectives.dll^
 -r:Alea.TypeProviders.GPUHelperProvider.dll^
 --target:library^
 Generate.fs