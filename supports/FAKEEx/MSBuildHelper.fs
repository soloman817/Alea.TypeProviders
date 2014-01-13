[<AutoOpen>]
module FakeEx.MSBuildHelper

open System
open System.Text
open System.IO
open Fake

let private errorLoggerParam = 
    let pathToLogger = (Uri(typedefof<MSBuildParams>.Assembly.CodeBase)).LocalPath
    [ TeamCityLoggerName; ErrorLoggerName ]
    |> List.map(fun a -> sprintf "/logger:%s,\"%s\"" a pathToLogger)
    |> fun lst -> String.Join(" ", lst)

let build setParams project =
    traceStartTask "MSBuild" project
    let args = MSBuildDefaults |> setParams |> serializeMSBuildParams        
    let args = args + " /nologo"
    let args = toParam project + " " + args + " " + errorLoggerParam
    //tracefn "Building project: %s\n  %s %s" project msBuildExe args
    tracefn "Building project: %s" project
    let exitCode =
        ExecProcess (fun info ->  
            info.FileName <- msBuildExe
            info.Arguments <- args) TimeSpan.MaxValue
    if exitCode <> 0 then
        if Diagnostics.Debugger.IsAttached then Diagnostics.Debugger.Break()
        let errors =
            System.Threading.Thread.Sleep(200) // wait for the file to write
            if File.Exists MsBuildLogger.ErrorLoggerFile then
                File.ReadAllLines(MsBuildLogger.ErrorLoggerFile) |> List.ofArray
            else []
        let errorMessage = sprintf "Building %s failed with exitcode %d." project exitCode
        raise (BuildException(errorMessage, errors))
    traceEndTask "MSBuild" project

/// Builds the given project files and collects the output files.
/// ## Parameters
///  - `outputpath` - If it is null or empty then the project settings are used.
///  - `targets` - A string with the target names which should be run by MSBuild.
///  - `properties` - A list with tuples of property name and property values.
let MSBuildWithProjectProperties outputPath (targets: string) (properties: string -> (string*string) list) projects = 
    let projects = projects |> Seq.toList
    let output = 
        if isNullOrEmpty outputPath then "" else
        outputPath
          |> FullName
          |> trimSeparator

    let properties = 
        if isNullOrEmpty output 
            then properties 
            else fun x -> ("OutputPath", output)::(properties x)

    let dependencies =
        projects 
            |> List.map getProjectReferences
            |> Set.unionMany

    let setBuildParam project projectParams = 
        { projectParams with
            Targets = targets |> split ';' 
            Properties = projectParams.Properties @ properties project
            Verbosity = Some MSBuildVerbosity.Minimal }

    projects
      |> List.filter (fun project -> not <| Set.contains project dependencies)
      |> List.iter (fun project -> build (setBuildParam project) project)
    
    // it makes no sense to output the root dir content here since it does not contain the build output
    if isNotNullOrEmpty output  then !! (outputPath @@ "/**/*.*") |> Seq.toList else []

/// Builds the given project files or solution files and collects the output files.
/// ## Parameters
///  - `outputpath` - If it is null or empty then the project settings are used.
///  - `targets` - A string with the target names which should be run by MSBuild.
///  - `properties` - A list with tuples of property name and property values.
let MSBuild outputPath targets properties = MSBuildWithProjectProperties outputPath targets (fun _ -> properties)

/// Builds the given project files or solution files and collects the output files.
/// ## Parameters
///  - `outputpath` - If it is null or empty then the project settings are used.
///  - `targets` - A string with the target names which should be run by MSBuild.
let MSBuildDebug outputPath targets = MSBuild outputPath targets ["Configuration","Debug"]

/// Builds the given project files or solution files and collects the output files.
/// ## Parameters
///  - `outputpath` - If it is null or empty then the project settings are used.
///  - `targets` - A string with the target names which should be run by MSBuild.
let MSBuildRelease outputPath targets = MSBuild outputPath targets ["Configuration","Release"]

/// Builds the given project files or solution files in release mode to the default outputs.
/// ## Parameters
///  - `targets` - A string with the target names which should be run by MSBuild.
let MSBuildWithDefaults targets = MSBuild null targets ["Configuration","Release"]

/// Builds the given project files or solution files in release mode and collects the output files.
/// ## Parameters
///  - `outputpath` - If it is null or empty then the project settings are used.
///  - `properties` - A list with tuples of property name and property values.
///  - `targets` - A string with the target names which should be run by MSBuild.
let MSBuildReleaseExt outputPath properties targets = 
    let properties = ("Configuration", "Release") :: properties; 
    MSBuild outputPath targets properties