#I "supports/FAKE/tools"
#r "FakeLib.dll"
#load "supports/FAKEEx/MSBuildHelper.fs"

// feature1 - 1
// feature1 - 2
// feature2 - 1
// feature2 - 2

open System
open System.IO
open System.Net
open System.Text.RegularExpressions
open Fake
open Fake.AssemblyInfoFile
open FakeEx

let now = DateTime.Now

let majorVersion = 1
let minorVersion = 0
let buildVersion = (now - DateTime(2012, 1, 1)).Days
let version = sprintf "%d.%d.%d" majorVersion minorVersion buildVersion
let company = "QuantAlea GmbH."
let copyright = sprintf "QuantAlea GmbH. 2010-%d" now.Year

let noTest = hasBuildParam "no-test"
let testNet40 = true
let testNet45 = hasBuildParam "test-net45"
let testX86 = hasBuildParam "test-x86"
let ignoreTestFail = hasBuildParam "ignore-test-fail"

[<Literal>]
let nuspecTemplate = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <metadata xmlns="http://schemas.microsoft.com/packaging/2010/07/nuspec.xsd">    
    <id>@project@</id>
    <version>@build.number@</version>
    <authors>@authors@</authors>
    <owners>@authors@</owners>
    <summary />
    __LICENSE_URL__
    <projectUrl>__PROJECT_URL__</projectUrl>
    <iconUrl>__ICON_URL__</iconUrl>
    <requireLicenseAcceptance>__REQUIRE_LICENSE_ACCEPTANCE__</requireLicenseAcceptance>
    <description>@description@</description>
    <releaseNotes>@releaseNotes@</releaseNotes>
    <copyright>@copyright@</copyright>    
    <tags>@tags@</tags>
    @dependencies@
  </metadata>
</package>"""

let writeNuSpecFile (path:string) (projectUrl:string) (iconUrl:string) (licenseUrl:string option) =
    let template = nuspecTemplate
    let template = Regex.Replace(template, "__PROJECT_URL__", projectUrl)
    let template = Regex.Replace(template, "__ICON_URL__", iconUrl)
    let template = licenseUrl |> function
        | None ->
            let template = Regex.Replace(template, "__LICENSE_URL__", "")
            Regex.Replace(template, "__REQUIRE_LICENSE_ACCEPTANCE__", "false")
        | Some licenseUrl ->
            let template = Regex.Replace(template, "__LICENSE_URL__", sprintf "<licenseUrl>%s</licenseUrl>" licenseUrl)
            Regex.Replace(template, "__REQUIRE_LICENSE_ACCEPTANCE__", "true")
    use file = new StreamWriter(path)
    file.Write(template)

let build() =
    let writeSolutionInfo() =
        let assemblyInfos =
            [ Attribute.Product "Alea.TypeProviders"
              Attribute.Company company
              Attribute.Copyright copyright
              Attribute.Version version
              Attribute.FileVersion version
              Attribute.InformationalVersion version ]

        CreateCSharpAssemblyInfo "SolutionInfo.cs" assemblyInfos
        CreateFSharpAssemblyInfo "SolutionInfo.fs" assemblyInfos

    let buildLibraries() =
        CleanDir "temp"
        !! ("src/*/*.fsproj")
        |> MSBuildRelease "temp" "Build"
        |> Log "Build-Output: "

    let buildSamplesAndTests() =
        !! ("tests/*/*.fsproj")
        |> MSBuildRelease "bin" "Build"
        |> Log "Build-Output: "

    writeSolutionInfo()
    buildLibraries()
    //buildSamplesAndTests()

let runOn (condition:bool) (name:string) (run:unit -> unit) =
    if condition then run()
    else tracefn "Skip %s." name

Target "Clean" (fun _ ->
    [ "bin"; "temp"; "output" ]
    |> CleanDirs
)

Target "Build" (fun _ ->
    build()
)

"Clean"
    ==> "Build"

RunTargetOrDefault "Build"
