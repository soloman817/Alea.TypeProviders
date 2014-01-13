copy /Y ..\..\..\..\bin\* .
copy /Y Generate.txt Generate.fs

set ProgFiles86Root=%ProgramFiles(x86)%
if not "%ProgFiles86Root%"=="" GOTO win64
set ProgFiles86Root=%ProgramFiles%
:win64

"%ProgFiles86Root%\Microsoft SDKs\F#\3.1\Framework\v4.0\fsc.exe"^
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