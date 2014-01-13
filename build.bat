@echo off
cls
rem supports\NuGet\NuGet.exe install FAKE -OutputDirectory supports -ExcludeVersion
rem supports\NuGet\NuGet.exe install FSharp.Formatting -OutputDirectory supports -ExcludeVersion -Prerelease
supports\FAKE\tools\FAKE.exe build.fsx %*
